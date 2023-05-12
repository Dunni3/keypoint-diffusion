from pathlib import Path
from time import time
import random
from collections import defaultdict
import argparse
import warnings
import yaml

from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB import PDBIO
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import QED
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from analysis.molecule_builder import build_molecule
import constants
import utils
import pickle

from data_processing.pdbbind_processing import rec_atom_featurizer, lig_atom_featurizer, Unparsable, build_receptor_graph
from utils import get_rec_atom_map

# dataset_info = dataset_params['bindingmoad']
# amino_acid_dict = dataset_info['aa_encoder']
# atom_dict = dataset_info['atom_encoder']
# atom_decoder = dataset_info['atom_decoder']


def element_fixer(element: str):

    if len(element) > 1:
        element = element[0] + element[1:].lower()
    
    return element


def read_label_file(csv_path):
    """
    Read BindingMOAD's label file
    Args:
        csv_path: path to 'every.csv'
    Returns:
        Nested dictionary with all ligands. First level: EC number,
            Second level: PDB ID, Third level: list of ligands. Each ligand is
            represented as a tuple (ligand name, validity, SMILES string)
    """
    ligand_dict = {}

    with open(csv_path, 'r') as f:
        for line in f.readlines():
            row = line.split(',')

            # new protein class
            if len(row[0]) > 0:
                curr_class = row[0]
                ligand_dict[curr_class] = {}
                continue

            # new protein
            if len(row[2]) > 0:
                curr_prot = row[2]
                ligand_dict[curr_class][curr_prot] = []
                continue

            # new small molecule
            if len(row[3]) > 0:
                ligand_dict[curr_class][curr_prot].append(
                    # (ligand name, validity, SMILES string)
                    [row[3], row[4], row[9]]
                )

    return ligand_dict


def compute_druglikeness(ligand_dict):
    """
    Computes RDKit's QED value and adds it to the dictionary
    Args:
        ligand_dict: nested ligand dictionary
    Returns:
        the same ligand dictionary with additional QED values
    """
    print("Computing QED values...")
    for p, m in tqdm([(p, m) for c in ligand_dict for p in ligand_dict[c]
                      for m in ligand_dict[c][p]]):
        mol = Chem.MolFromSmiles(m[2])
        if mol is None:
            mol_id = f'{p}_{m}'
            warnings.warn(f"Could not construct molecule {mol_id} from SMILES "
                          f"string '{m[2]}'")
            continue
        m.append(QED.qed(mol))
    return ligand_dict


def filter_and_flatten(ligand_dict, qed_thresh, max_occurences, seed):

    filtered_examples = []
    all_examples = [(c, p, m) for c in ligand_dict for p in ligand_dict[c]
                    for m in ligand_dict[c][p]]

    # shuffle to select random examples of ligands that occur more than
    # max_occurences times
    random.seed(seed)
    random.shuffle(all_examples)

    ligand_name_counter = defaultdict(int)
    print("Filtering examples...")
    for c, p, m in tqdm(all_examples):

        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
        if m[1] == 'valid' and len(m) > 3 and m[3] > qed_thresh:
            if ligand_name_counter[ligand_name] < max_occurences:
                filtered_examples.append(
                    (c, p, m)
                )
                ligand_name_counter[ligand_name] += 1

    return filtered_examples


def split_by_ec_number(data_list, n_val, n_test, ec_level=1):
    """
    Split dataset into training, validation and test sets based on EC numbers
    https://en.wikipedia.org/wiki/Enzyme_Commission_number
    Args:
        data_list: list of ligands
        n_val: number of validation examples
        n_test: number of test examples
        ec_level: level in the EC numbering hierarchy at which the split is
            made, i.e. items with matching EC numbers at this level are put in
            the same set
    Returns:
        dictionary with keys 'train', 'val', and 'test'
    """

    examples_per_class = defaultdict(int)
    for c, p, m in data_list:
        c_sub = '.'.join(c.split('.')[:ec_level])
        examples_per_class[c_sub] += 1

    assert sum(examples_per_class.values()) == len(data_list)

    # split ec numbers
    val_classes = set()
    for c, num in sorted(examples_per_class.items(), key=lambda x: x[1],
                         reverse=True):
        if sum([examples_per_class[x] for x in val_classes]) + num <= n_val:
            val_classes.add(c)

    test_classes = set()
    for c, num in sorted(examples_per_class.items(), key=lambda x: x[1],
                         reverse=True):
        # skip classes already used in the validation set
        if c in val_classes:
            continue
        if sum([examples_per_class[x] for x in test_classes]) + num <= n_test:
            test_classes.add(c)

    # remaining classes belong to test set
    train_classes = {x for x in examples_per_class if
                     x not in val_classes and x not in test_classes}

    # create separate lists of examples
    data_split = {}
    data_split['train'] = [x for x in data_list if '.'.join(
        x[0].split('.')[:ec_level]) in train_classes]
    data_split['val'] = [x for x in data_list if '.'.join(
        x[0].split('.')[:ec_level]) in val_classes]
    data_split['test'] = [x for x in data_list if '.'.join(
        x[0].split('.')[:ec_level]) in test_classes]

    assert len(data_split['train']) + len(data_split['val']) + \
           len(data_split['test']) == len(data_list)

    return data_split


def ligand_list_to_dict(ligand_list):
    out_dict = defaultdict(list)
    for _, p, m in ligand_list:
        out_dict[p].append(m)
    return out_dict


def process_ligand_and_pocket(pdb_struct, ligand_name, ligand_chain,
                              ligand_resi, dist_cutoff, ca_only,
                              compute_quaternion=False):
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a for a in ligand.get_atoms()
                 if (a.element.capitalize() in atom_dict or a.element != 'H')]
    lig_coords = np.array([a.get_coord() for a in lig_atoms])

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.element.capitalize()]).squeeze()
            for a in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(
            f'Ligand atom {e} not in atom dict ({pdbfile}, '
            f'{ligand_name}:{ligand_chain}:{ligand_resi})')

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    # Compute transform of the canonical reference frame
    n_xyz = np.array([res['N'].get_coord() for res in pocket_residues])
    ca_xyz = np.array([res['CA'].get_coord() for res in pocket_residues])
    c_xyz = np.array([res['C'].get_coord() for res in pocket_residues])

    if compute_quaternion:
        quaternion, c_alpha = get_bb_transform(n_xyz, ca_xyz, c_xyz)
        if np.any(np.isnan(quaternion)):
            raise ValueError(
                f'Invalid value in quaternion ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    else:
        c_alpha = ca_xyz

    if ca_only:
        pocket_coords = c_alpha
        try:
            pocket_one_hot = np.stack([
                np.eye(1, len(amino_acid_dict),
                       amino_acid_dict[three_to_one(res.get_resname())]).squeeze()
                for res in pocket_residues])
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    else:
        pocket_atoms = [a for res in pocket_residues for a in res.get_atoms()
                        if (a.element.capitalize() in atom_dict or a.element != 'H')]
        pocket_coords = np.array([a.get_coord() for a in pocket_atoms])
        try:
            pocket_one_hot = np.stack([
                np.eye(1, len(atom_dict), atom_dict[a.element.capitalize()]).squeeze()
                for a in pocket_atoms
            ])
        except KeyError as e:
            raise KeyError(
                f'Pocket atom {e} not in atom dict ({pdbfile}, '
                f'{ligand_name}:{ligand_chain}:{ligand_resi})')

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]

    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }
    pocket_data = {
        'pocket_ca': pocket_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids,
    }
    if compute_quaternion:
        pocket_data['pocket_quaternion'] = quaternion
    return ligand_data, pocket_data

def process_ligand_and_pocket_new(pdb_struct, ligand_name, ligand_chain, ligand_resi,
                                  rec_element_map, lig_element_map,
                                  receptor_k: int, pocket_edge_algorithm: str, 
                                  dist_cutoff: float, remove_hydrogen: bool = True):
    
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise Unparsable(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    ligand = residues[ligand_resi]
    assert ligand.get_resname() == ligand_name, \
        f"{ligand.get_resname()} != {ligand_name}"

    lig_atoms = [a for a in ligand.get_atoms()]

    # remove hydrogens if required
    if remove_hydrogen:
        lig_atoms = [ a for a in lig_atoms if a.element.capitalize() != 'H' ]

    # get ligand coordinates and elements
    lig_coords = np.array([a.get_coord() for a in lig_atoms])
    lig_elements = [ element_fixer(a.element) for a in lig_atoms ]

    # one-hot encode, throw error if unsupported atom type found
    lig_atom_features, other_atoms_mask = lig_atom_featurizer(lig_element_map, atom_elements=lig_elements)
    if other_atoms_mask.sum() != 0:
        raise Unparsable(f'unsupported atoms found: { np.array(lig_elements)[other_atoms_mask] }')


    # get residues which constitute the binding pocket
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():

        # get atomic coordinates of residue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])

        # check if residue is interacting with protein
        is_residue = is_aa(residue.get_resname(), standard=True)
        if not is_residue:
            continue
        min_rl_dist = cdist(lig_coords, res_coords).min()
        if min_rl_dist < dist_cutoff:
            pocket_residues.append(residue)

    if len(pocket_residues) == 0:
        raise Unparsable(f'no valid pocket residues found in {pdbfile}: {ligand_name}:{ligand_chain}:{ligand_resi})', )

    pocket_atoms = [a for res in pocket_residues for a in res.get_atoms() ]
    if remove_hydrogen:
        pocket_atoms = [ a for a in pocket_atoms if a.element != "H" ]

    pocket_coords = torch.tensor(np.array([a.get_coord() for a in pocket_atoms]))
    pocket_elements = np.array([ element_fixer(a.element) for a in pocket_atoms ])
    pocket_atom_features, other_atoms_mask = rec_atom_featurizer(rec_element_map, protein_atom_elements=pocket_elements)
    pocket_atom_features = torch.tensor(pocket_atom_features).bool()

    # remove other atoms from pocket
    pocket_coords = pocket_coords[~other_atoms_mask]
    pocket_atom_features = pocket_atom_features[~other_atoms_mask]

    rec_graph = build_receptor_graph(pocket_coords, pocket_atom_features, k=receptor_k, edge_algorithm=pocket_edge_algorithm)


    lig_coords = torch.tensor(lig_coords)

    return rec_graph, lig_coords, lig_atom_features


def compute_smiles(lig_pos, lig_feat, lig_decoder):
    atom_types = [ lig_decoder[x] for x in torch.argmax(lig_feat.int(), dim=1).tolist() ]
    mol = build_molecule(lig_pos, atom_types, sanitize=True)

    if mol is None:
        return None
    
    smi = Chem.MolToSmiles(mol)
    return smi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--config_file', type=Path, required=True)

    # parser.add_argument('--qed_thresh', type=float, default=0.3)
    # parser.add_argument('--max_occurences', type=int, default=50)
    # parser.add_argument('--num_val', type=int, default=300)
    # parser.add_argument('--num_test', type=int, default=300)
    # parser.add_argument('--dist_cutoff', type=float, default=8.0)
    # parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    # parser.add_argument('--make_split', action='store_true')


    args = parser.parse_args()

    pdbdir = args.data_dir / 'BindingMOAD_2020'


    # load dataset config
    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config = config_dict['dataset']

    # construct atom typing maps
    rec_element_map, lig_element_map = get_rec_atom_map(dataset_config)
    lig_decoder = { v:k for k,v in lig_element_map.items() }

    processed_dir = Path(dataset_config['location'])
    processed_dir.mkdir(exist_ok=True, parents=True)

    # code for making dataset splits from DiffSBDD's data processing script .... we just use their splits for now
    # if args.make_split:
    #     # Process the label file
    #     csv_path = args.basedir / 'every.csv'
    #     ligand_dict = read_label_file(csv_path)
    #     ligand_dict = compute_druglikeness(ligand_dict)
    #     filtered_examples = filter_and_flatten(
    #         ligand_dict, args.qed_thresh, args.max_occurences, args.random_seed)
    #     print(f'{len(filtered_examples)} examples after filtering')

    #     # Make data split
    #     data_split = split_by_ec_number(filtered_examples, args.num_val,
    #                                     args.num_test)

    # else:

    # Use precomputed data split
    data_split = {}
    for split in ['test', 'val', 'train']:
        with open(args.data_dir / f'moad_{split}.txt', 'r') as f:
            pocket_ids = f.read().split(',')
        # (ec-number, protein, molecule tuple)

        # truncate dataset if necessary (only used for debugging purposes)
        if dataset_config['dataset_size'] is not None and len(pocket_ids) > dataset_config['dataset_size']:
            pocket_ids = pocket_ids[:dataset_config['dataset_size']]

        data_split[split] = [(None, x.split('_')[0][:4], (x.split('_')[1],))
                        for x in pocket_ids]

    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    # Read and process PDB files
    n_samples_after = {}
    for split in data_split.keys():
        print(f'processing {split} split', flush=True)

        count = 0
        

        data = defaultdict(list)
        ligand_size_counter = defaultdict(int)
        atom_type_counts = None
        smiles = set()

        pdb_sdf_dir = processed_dir / f'{split}_structures'
        pdb_sdf_dir = pdb_sdf_dir.resolve()
        pdb_sdf_dir.mkdir(exist_ok=True)

        n_tot = len(data_split[split])
        pair_dict = ligand_list_to_dict(data_split[split])

        tic = time()
        num_failed = 0
        with tqdm(total=n_tot) as pbar:
            for p in pair_dict:

                pdb_successful = set()

                # try all available .bio files
                for pdbfile in sorted(pdbdir.glob(f"{p.lower()}.bio*")):

                    # Skip if all ligands have been processed already
                    if len(pair_dict[p]) == len(pdb_successful):
                        continue

                    # figure out the output filepath of this receptor should we need to save it later on
                    pdb_file_out = Path(pdb_sdf_dir, f'{p}_{pdbfile.suffix[1:]}.pdb')

                    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
                    struct_copy = pdb_struct.copy()

                    n_bio_successful = 0
                    for m in pair_dict[p]:

                        # Skip already processed ligand
                        if m[0] in pdb_successful:
                            continue

                        ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                        ligand_resi = int(ligand_resi)

                        try:
                            rec_graph, lig_atom_positions, lig_atom_features = process_ligand_and_pocket_new(pdb_struct, 
                                                        ligand_name, 
                                                        ligand_chain, 
                                                        ligand_resi,
                                                        rec_element_map=rec_element_map,
                                                        lig_element_map=lig_element_map,
                                                        receptor_k=dataset_config['receptor_k'],
                                                        pocket_edge_algorithm=dataset_config['pocket_edge_algorithm'], 
                                                        dist_cutoff=dataset_config['pocket_cutoff'], 
                                                        remove_hydrogen=dataset_config['remove_hydrogen'])
                        except Unparsable as e:
                            print(e)
                            continue

                        count += 1

                        pdb_successful.add(m[0])
                        n_bio_successful += 1

                        # write ligand to sdf file if this is the validation or test set
                        # if we are in the validation/test set, and is not possible to extract the ligand, we want to exclude this ligand completely
                        if split in {'val', 'test'}:
                            # remove ligand from receptor
                            try:
                                struct_copy[0][ligand_chain].detach_child((f'H_{ligand_name}', ligand_resi, ' '))
                            except KeyError as e:
                                warnings.warn(f"Could not find ligand {(f'H_{ligand_name}', ligand_resi, ' ')} in {pdbfile}")
                                continue

                            # Create SDF file
                            atom_types = [ lig_decoder[x] for x in torch.argmax(lig_atom_features.int(), dim=1).tolist() ]
                            xyz_file = Path(pdb_sdf_dir, 'tmp.xyz')
                            xyz_file_str = utils.write_xyz_file(lig_atom_positions, atom_types)

                            obConversion = openbabel.OBConversion()
                            obConversion.SetInAndOutFormats("xyz", "sdf")
                            mol = openbabel.OBMol()
                            obConversion.ReadString(mol, xyz_file_str)

                            name = f"{p}_{pdbfile.suffix[1:]}_{m[0]}"
                            sdf_file = Path(pdb_sdf_dir, f'{name}.sdf')
                            obConversion.WriteFile(mol, str(sdf_file))

                        # update counts of atom types
                        if atom_type_counts is None:
                            atom_type_counts = lig_atom_features.sum(dim=0)
                        else:
                            atom_type_counts += lig_atom_features.sum(dim=0)

                        # record ligand size
                        ligand_size_counter[lig_atom_positions.shape[0]] += 1

                        # compute/record smiles
                        smi = compute_smiles(lig_atom_positions, lig_atom_features, lig_decoder)
                        if smi is not None:
                            smiles.add(smi)

                        # add graphs, ligand positions, and ligand features to the dataset
                        data['receptor_graph'].append(rec_graph)
                        data['lig_atom_positions'].append(lig_atom_positions)
                        data['lig_atom_features'].append(lig_atom_features)
                        if split in {'val', 'test'}:
                            data['rec_files'].append(str(pdb_file_out))
                            data['lig_files'].append(str(sdf_file))


                    if split in {'val', 'test'} and n_bio_successful > 0:
                        # create receptor PDB file
                        # pdb_file_out = Path(pdb_sdf_dir, f'{p}_{pdbfile.suffix[1:]}.pdb')
                        io = PDBIO()
                        io.set_structure(struct_copy)
                        io.save(str(pdb_file_out))


                pbar.update(len(pair_dict[p]))
                num_failed += (len(pair_dict[p]) - len(pdb_successful))
                pbar.set_description(f'#failed: {num_failed}')


        # save data for this split
        data_filepath = processed_dir / f'{split}.pkl'
        with open(data_filepath, 'wb') as f:
            pickle.dump(data, f)

        # compute/save atom type counts
        type_counts_file = processed_dir / f'{split}_type_counts.pkl'
        with open(type_counts_file, 'wb') as f:
            pickle.dump(atom_type_counts, f)

        # save filenames
        filenames = processed_dir / f'{split}_filenames.pkl'
        with open(filenames, 'wb') as f:
            pickle.dump({'rec_files': data['rec_files'], 'lig_files': data['lig_files']}, f)

        # save ligand size counts
        lig_size_file = processed_dir / f'{split}_ligand_sizes.pkl'
        with open(lig_size_file, 'wb') as f:
            pickle.dump(ligand_size_counter, f)

        # save smiles
        smiles_file = processed_dir / f'{split}_smiles.pkl'
        with open(smiles_file, 'wb') as f:
            pickle.dump(smiles, f)

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

from data_processing.pdbbind_processing import rec_atom_featurizer, lig_atom_featurizer, Unparsable, build_receptor_graph, get_interface_points, InterfacePointException, build_initial_complex_graph
from utils import get_rec_atom_map


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

def ligand_list_to_dict(ligand_list):
    out_dict = defaultdict(list)
    for _, p, m in ligand_list:
        out_dict[p].append(m)
    return out_dict

def process_ligand_and_pocket(pdb_struct, ligand_name, ligand_chain, ligand_resi,
                                  rec_element_map, lig_element_map,
                                  receptor_k: int, pocket_edge_algorithm: str, 
                                  dist_cutoff: float, remove_hydrogen: bool = True):
    
    try:
        residues = {obj.id[1]: obj for obj in
                    pdb_struct[0][ligand_chain].get_residues()}
    except KeyError as e:
        raise Unparsable(f'Chain {e} not found ({pdbfile}, '
                       f'{ligand_name}:{ligand_chain}:{ligand_resi})')
    try:
        ligand = residues[ligand_resi]
    except KeyError:
        raise Unparsable('ligand residue index not found')
    try:
        assert ligand.get_resname() == ligand_name, \
            f"{ligand.get_resname()} != {ligand_name}"
    except AssertionError:
        raise Unparsable('ligand resname assertion failed')

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
    
    # drop "other atoms" column
    lig_atom_features = lig_atom_features[:, :-1]


    # make ligand data into torch tensors
    lig_coords = torch.tensor(lig_coords, dtype=torch.float32)
    lig_atom_features = torch.tensor(lig_atom_features, dtype=torch.float32)

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
        if min_rl_dist < pocket_cutoff:
            pocket_residues.append(residue)

    if len(pocket_residues) == 0:
        raise Unparsable(f'no valid pocket residues found in {pdbfile}: {ligand_name}:{ligand_chain}:{ligand_resi})', )

    if remove_hydrogen:
        atom_filter = lambda a: a.element != "H"
    else:
        atom_filter = lambda a: True

    pocket_atomres = [(a, res) for res in pocket_residues for a in res.get_atoms() if atom_filter(a) ]
    pocket_atoms, atom_residues = list(map(list, zip(*pocket_atomres)))
    res_to_idx = { res:i for i, res in enumerate(pocket_residues) }
    pocket_res_idx = list(map(lambda res: res_to_idx[res], atom_residues)) #  list containing the residue of every atom using integers to index pocket residues
    pocket_res_idx = torch.tensor(pocket_res_idx)

    pocket_coords = torch.tensor(np.array([a.get_coord() for a in pocket_atoms]))
    pocket_elements = np.array([ element_fixer(a.element) for a in pocket_atoms ])
    pocket_atom_features, other_atoms_mask = rec_atom_featurizer(rec_element_map, protein_atom_elements=pocket_elements)
    pocket_atom_features = torch.tensor(pocket_atom_features).bool()

    # remove other atoms from pocket
    pocket_coords = pocket_coords[~other_atoms_mask]
    pocket_atom_features = pocket_atom_features[~other_atoms_mask]

    # rec_graph = build_receptor_graph(pocket_coords, pocket_atom_features, k=receptor_k, edge_algorithm=pocket_edge_algorithm)
    complex_graph = build_initial_complex_graph(pocket_coords, pocket_atom_features, lig_coords, lig_atom_features, pocket_res_idx, n_keypoints=n_keypoints, cutoffs=graph_cutoffs)


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
    graph_config = config_dict['graph']

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
        pdb_sdf_dir = pdb_sdf_dir
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
                            rec_graph, lig_atom_positions, lig_atom_features, interface_points = process_ligand_and_pocket(pdb_struct, 
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
                        except InterfacePointException as e:
                            print('interface point exception occured', flush=True)
                            print(e)
                            print(e.original_exception)
                            continue
                        except Exception as e:
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
                        data['interface_points'].append(interface_points)
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

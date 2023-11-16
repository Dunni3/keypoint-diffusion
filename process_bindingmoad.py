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
from torch.nn.functional import one_hot
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1, is_aa
from Bio.PDB import PDBIO
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import QED
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from analysis.molecule_builder import build_molecule
from constants import aa_to_idx
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
                                  ip_dist_threshold: float, ip_exclusion_threshold: float, 
                                  pocket_cutoff: float, remove_hydrogen: bool = True, ca_only: bool = False):
    
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

    pocket_atomres = []
    for res in pocket_residues:

        if ca_only:
            atom_list = [ res['CA'] ]
        else:
            atom_list = res.get_atoms()

        pocket_atomres.extend([(a, res) for a in atom_list if atom_filter(a) ])

    pocket_atoms, atom_residues = list(map(list, zip(*pocket_atomres)))
    res_to_idx = { res:i for i, res in enumerate(pocket_residues) }
    pocket_res_idx = list(map(lambda res: res_to_idx[res], atom_residues)) #  list containing the residue of every atom using integers to index pocket residues
    pocket_res_idx = torch.tensor(pocket_res_idx)

    pocket_coords = torch.tensor(np.array([a.get_coord() for a in pocket_atoms]))
    pocket_elements = np.array([ element_fixer(a.element) for a in pocket_atoms ])

    if ca_only:

        try:
            # get single-letter amino acid code for each residue
            res_chars = [ protein_letters_3to1[res.get_resname()] for res in pocket_residues ]

            # convert residue characters to indices
            res_idx = [ aa_to_idx[res] for res in res_chars ]
            res_idx = torch.tensor(res_idx)
        except KeyError as e:
            raise Unparsable(f'unsupported residue type found: {[ res.get_resname() for res in pocket_residues ]}')

        # one-hot encode residue types
        pocket_atom_features = one_hot(res_idx, num_classes=len(aa_to_idx)).bool()

        # create an empty other_atoms_mask
        other_atoms_mask = torch.zeros(pocket_atom_features.shape[0], dtype=torch.bool)

    else:
        pocket_atom_features, other_atoms_mask = rec_atom_featurizer(rec_element_map, protein_atom_elements=pocket_elements)
        pocket_atom_features = torch.tensor(pocket_atom_features).bool()

    # remove other atoms from pocket
    pocket_coords = pocket_coords[~other_atoms_mask]
    pocket_atom_features = pocket_atom_features[~other_atoms_mask]

    # compute interface points
    if ca_only:
        # the ca_only dataset is strictly for a baseline where we are not learning a keypoint representation, and instead 
        # just using the CA atoms as the receptor reperesentation for diffusion
        # therefore, we don't need interface points because we are not learning a keypoint representation
        interface_points = torch.zeros(0,3)
    else:
        try:
            interface_points = get_interface_points(lig_coords, pocket_coords, distance_threshold=ip_dist_threshold, exclusion_threshold=ip_exclusion_threshold)
        except Exception as e:
            raise InterfacePointException(e)

    return pocket_coords, pocket_atom_features, lig_coords, lig_atom_features, pocket_res_idx, interface_points


def compute_smiles(lig_pos, lig_feat, lig_decoder):
    atom_types = [ lig_decoder[x] for x in torch.argmax(lig_feat.int(), dim=1).tolist() ]
    mol = build_molecule(lig_pos, atom_types, sanitize=True)

    if mol is None:
        return None
    
    smi = Chem.MolToSmiles(mol)
    return smi

def get_n_nodes_dist(lig_rec_size_counter: defaultdict, smooth_sigma=1):
    # Joint distribution of ligand's and pocket's number of nodes

    observed_rec_num_nodes, observed_lig_num_nodes = list(zip(*lig_rec_size_counter.keys()))


    # idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    # idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)

    # joint_histogram = np.zeros((np.max(lig_num_nodes) + 1,
    #                             np.max(rec_num_nodes) + 1))

    # for nlig, npocket in zip(lig_num_nodes, rec_num_nodes):
    #     joint_histogram[nlig, npocket] += 1

    joint_histogram = np.zeros((
        np.max(observed_rec_num_nodes) - np.min(observed_rec_num_nodes) + 1,
        np.max(observed_lig_num_nodes) - np.min(observed_lig_num_nodes) + 1
    ))

    rec_idx_val_map = np.arange(np.min(observed_rec_num_nodes), np.max(observed_rec_num_nodes) + 1)
    rec_val_idx_map = { val:idx for idx, val in enumerate(rec_idx_val_map) }

    lig_idx_val_map = np.arange(np.min(observed_lig_num_nodes), np.max(observed_lig_num_nodes) + 1)
    lig_val_idx_map = { val:idx for idx, val in enumerate(lig_idx_val_map) }

    for mol_sizes, count in lig_rec_size_counter.items():
        rec_n_atoms, lig_n_atoms = mol_sizes
        rec_idx = rec_val_idx_map[rec_n_atoms]
        lig_idx = lig_val_idx_map[lig_n_atoms]
        joint_histogram[rec_idx, lig_idx] += count

    joint_histogram = joint_histogram / joint_histogram.sum()

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

        joint_histogram = joint_histogram / joint_histogram.sum()

    rec_bounds = (rec_idx_val_map[0], rec_idx_val_map[-1])
    lig_bounds = (lig_idx_val_map[0], lig_idx_val_map[-1])

    return joint_histogram, rec_bounds, lig_bounds


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

    # determine if we are using a Ca-only representation of the receptor
    try:
        ca_only: bool = dataset_config['ca_only']
    except KeyError:
        ca_only = False

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
        lig_rec_size_counter = defaultdict(int)
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
                            rec_pos, rec_feat, lig_pos, lig_feat, rec_res_idx, interface_points = process_ligand_and_pocket(pdb_struct, 
                                                        ligand_name, 
                                                        ligand_chain, 
                                                        ligand_resi,
                                                        rec_element_map=rec_element_map,
                                                        lig_element_map=lig_element_map,
                                                        ip_dist_threshold=dataset_config['interface_distance_threshold'],
                                                        ip_exclusion_threshold=dataset_config['interface_exclusion_threshold'], 
                                                        pocket_cutoff=dataset_config['pocket_cutoff'], 
                                                        remove_hydrogen=dataset_config['remove_hydrogen'],
                                                        ca_only=ca_only)
                        except Unparsable as e:
                            print(e)
                            continue
                        except InterfacePointException as e:
                            print('interface point exception occured', flush=True)
                            print(e)
                            print(e.original_exception)
                            continue
                        except Exception as e:
                            raise e
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
                            atom_types = [ lig_decoder[x] for x in torch.argmax(lig_feat.int(), dim=1).tolist() ]
                            xyz_file = Path(pdb_sdf_dir, 'tmp.xyz')
                            xyz_file_str = utils.write_xyz_file(lig_pos, atom_types)

                            obConversion = openbabel.OBConversion()
                            obConversion.SetInAndOutFormats("xyz", "sdf")
                            mol = openbabel.OBMol()
                            obConversion.ReadString(mol, xyz_file_str)

                            name = f"{p}_{pdbfile.suffix[1:]}_{m[0]}"
                            sdf_file = Path(pdb_sdf_dir, f'{name}.sdf')
                            obConversion.WriteFile(mol, str(sdf_file))

                        # update counts of atom types
                        if atom_type_counts is None:
                            atom_type_counts = lig_feat.sum(dim=0)
                        else:
                            atom_type_counts += lig_feat.sum(dim=0)

                        # record ligand size and pocket size
                        size_counter_key = (rec_pos.shape[0], lig_pos.shape[0])
                        lig_rec_size_counter[size_counter_key] += 1

                        # compute/record smiles
                        smi = compute_smiles(lig_pos, lig_feat, lig_decoder)
                        if smi is not None:
                            smiles.add(smi)

                        # add graphs, ligand positions, and ligand features to the dataset
                        data['lig_pos'].append(lig_pos)
                        data['lig_feat'].append(lig_feat)
                        data['rec_pos'].append(rec_pos)
                        data['rec_feat'].append(rec_feat)
                        data['rec_res_idx'].append(rec_res_idx)
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


        # concatenate data for this split and compute idx lookups so we can split data examples back out

        n_graphs = len(data['lig_pos'])
        rec_segments = torch.zeros(n_graphs+1, dtype=int)
        lig_segments = rec_segments.clone()
        ip_segments = rec_segments.clone()
        rec_segments[1:] = torch.tensor([ x.shape[0] for x in data['rec_pos'] ], dtype=int)
        lig_segments[1:] = torch.tensor([ x.shape[0] for x in data['lig_pos'] ], dtype=int)
        ip_segments[1:] = torch.tensor([ x.shape[0] for x in data['interface_points'] ], dtype=int)

        for key in data:
            if 'files' in key:
                continue
            data[key] = torch.concatenate(data[key], dim=0)

        data['rec_segments'] = torch.cumsum(rec_segments, dim=0)
        data['lig_segments'] = torch.cumsum(lig_segments, dim=0)
        data['ip_segments'] = torch.cumsum(ip_segments, dim=0)


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

        # save joint distribution of ligand and pocket sizes
        joint_dist_file = processed_dir / f'{split}_n_node_joint_dist.pkl'
        joint_dist_data = get_n_nodes_dist(lig_rec_size_counter, smooth_sigma=1)
        with open(joint_dist_file, 'wb') as f:
            pickle.dump(joint_dist_data, f)

        # save smiles
        smiles_file = processed_dir / f'{split}_smiles.pkl'
        with open(smiles_file, 'wb') as f:
            pickle.dump(smiles, f)

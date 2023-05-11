import argparse
import yaml
import torch
from pathlib import Path
import prody
from typing import Dict
import dgl
import pickle
import numpy as np
from rdkit import Chem
from collections import defaultdict

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights, center_complex, Unparsable)
from utils import get_rec_atom_map


prody.confProDy(verbosity='none')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='path to crossdocked directory')
    p.add_argument('--index_file', type=str, required=True, help='file containing train/test splits')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/dev_config.yml')
    p.add_argument('--skip_train', action='store_true')

    args = p.parse_args()

    args.data_dir = Path(args.data_dir)
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)

    return args, config_dict

if __name__ == "__main__":
    args, config = parse_args()

    dataset_config = config['dataset']

    # determine dataset size 
    if dataset_config['dataset_size'] is None:
        dataset_size = np.inf
    else:
        dataset_size = dataset_config['dataset_size']

    # create output directory if necessary
    output_dir = Path(dataset_config['location'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # construct atom typing maps
    rec_element_map, lig_element_map = get_rec_atom_map(dataset_config)

    # save pickled arguments/dataset_config to output dir
    # TODO: this throws an exception, i don't know why, too tired to deal with it right now
    # args_file = args.output_dir / 'args.pkl'
    # with open(args_file, 'wb') as f:
    #     pickle.dump([vars(args), dataset_config], f)

    # load dataset index
    dataset_index = torch.load(args.index_file)
    
    for split_key in dataset_index:

        if split_key == "train" and args.skip_train:
            continue

        dataset_idx = 0
        data = defaultdict(list)
        ligand_size_counter = defaultdict(int)
        smiles = set()
        atom_type_counts = None
        for pair_idx, input_pair in enumerate(dataset_index[split_key]):

            if pair_idx % 10000 == 0:
                print(f'{pair_idx} complexes processed', flush=True)

            # break early if the dataset size is limited
            # this is only for working with truncated datasets for debugging/development purposes
            if pair_idx >= dataset_size:
                break

            # get filepath of receptor and ligand file
            rec_file = args.data_dir / input_pair[0]
            lig_file = args.data_dir / input_pair[1]

            # get atoms from receptor file
            try:
                rec_atoms: prody.Selection = parse_protein(rec_file, remove_hydrogen=dataset_config['remove_hydrogen'])
            except Unparsable:
                print(f'unparsable file: {rec_file}')
                continue

            # get rdkit molecule from ligand, as well as atom positions/features
            try:
                ligand, lig_atom_positions, lig_atom_features = parse_ligand(lig_file, element_map=lig_element_map, 
                    remove_hydrogen=dataset_config['remove_hydrogen'])
            except Unparsable:
                print(f'ligand has unsupported atom types: {lig_file}')
                continue


            # skip ligands smaller than minimum ligand size
            if lig_atom_positions.shape[0] < dataset_config['min_ligand_atoms']:
                continue

            # get all protein atoms that form the binding pocket
            pocket_atom_positions, pocket_atom_features, pocket_atom_mask \
            = get_pocket_atoms(rec_atoms, 
                lig_atom_positions, 
                box_padding=dataset_config["lig_box_padding"], 
                pocket_cutoff=dataset_config["pocket_cutoff"], 
                element_map=rec_element_map)

            # TODO: sometimes (rarely) pocket_atom_positions is an empty tensor. I'm just going to skip these instances
            # but for future reference, here is a receptor for which this happens:
            # rec_file = /home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10/RDM1_ARATH_7_163_0/2q3t_A_rec_2q3t_cps_lig_tt_docked_278_pocket10.pdb
            # lig_file = /home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10/RDM1_ARATH_7_163_0/2q3t_A_rec_2q3t_cps_lig_tt_docked_278.sdf
            if pocket_atom_positions.shape[0] == 0:
                continue

            # get boltzmann probabilities for OT loss
            # TODO: implement computing boltzmann probabilities for OT loss
            if dataset_config['use_boltzmann_ot']:
                ot_loss_boltzmann_weights = get_ot_loss_weights()

            # build receptor graph
            receptor_graph = build_receptor_graph(pocket_atom_positions,
                pocket_atom_features, 
                dataset_config['receptor_k'], 
                dataset_config['pocket_edge_algorithm'])

            # record data
            data['receptor_graph'].append(receptor_graph)
            data['lig_atom_positions'].append(lig_atom_positions)
            data['lig_atom_features'].append(lig_atom_features)
            data['rec_files'].append(str(rec_file))
            data['lig_files'].append(str(lig_file))

            # compute/record smiles
            try:
                smi = Chem.MolToSmiles(ligand)
            except:
                print('failed to convert to smiles', flush=True)
                
            if smi is not None:
                smiles.add(smi)

            # update atom counts
            # NOTE: we are assuming that ligand atom features are strictly one-hots of atom type so.
            # this might not always be the case, for example, maybe we want partial charges to be an atom-features
            # but for now, we're only using atom types, so ligand features == atom type one-hots
            if atom_type_counts is None:
                atom_type_counts = lig_atom_features.sum(dim=0)
            else:
                atom_type_counts += lig_atom_features.sum(dim=0)

            # record ligand size
            ligand_size_counter[lig_atom_positions.shape[0]] += 1

            # increment dataset_idx
            dataset_idx += 1

        # compute/save atom type counts
        type_counts_file = output_dir / f'{split_key}_type_counts.pkl'
        with open(type_counts_file, 'wb') as f:
            pickle.dump(atom_type_counts, f)

        # save data for this split
        data_filepath = output_dir / f'{split_key}.pkl'
        with open(data_filepath, 'wb') as f:
            pickle.dump(data, f)

        filenames = output_dir / f'{split_key}_filenames.pkl'
        with open(filenames, 'wb') as f:
            pickle.dump({'rec_files': data['rec_files'], 'lig_files': data['lig_files']}, f)

        # save ligand size counts
        lig_size_file = output_dir / f'{split_key}_ligand_sizes.pkl'
        with open(lig_size_file, 'wb') as f:
            pickle.dump(ligand_size_counter, f)

        # save smiles
        smiles_file = output_dir / f'{split_key}_smiles.pkl'
        with open(smiles_file, 'wb') as f:
            pickle.dump(smiles, f)
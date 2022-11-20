import argparse
import yaml
import torch
from pathlib import Path
import prody
from typing import Dict
import dgl
import pickle

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='path to crossdocked directory')
    p.add_argument('--output_dir', type=str, required=True, help='path to crossdocked directory')
    p.add_argument('--index_file', type=str, required=True, help='file containing train/test splits')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/dev_config.yml')

    args = p.parse_args()

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)

    return args, config_dict

if __name__ == "__main__":
    args, config = parse_args()

    dataset_config = config['dataset_config']

    # create output directory if necessary
    args.output_dir.mkdir(exist_ok=True)

    # construct atom typing maps
    rec_elements = dataset_config['rec_elements']
    rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(rec_elements) }

    lig_elements = dataset_config['lig_elements']
    lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(lig_elements) }

    # save pickled arguments/dataset_config to output dir
    # TODO: this throws an exception, i don't know why, too tired to deal with it right now
    # args_file = args.output_dir / 'args.pkl'
    # with open(args_file, 'wb') as f:
    #     pickle.dump([vars(args), dataset_config], f)

    # load dataset index
    dataset_index = torch.load(args.index_file)
    
    for split_key in dataset_index:

        for pair_idx, input_pair in enumerate(dataset_index[split_key]):

            # break early if the dataset size is limited
            # this is only for working with truncated datasets for debugging/development purposes
            if pair_idx > dataset_config['dataset_size']:
                break

            # get filepath of receptor and ligand file
            rec_file = args.data_dir / input_pair[0]
            lig_file = args.data_dir / input_pair[1]

            # get atoms from receptor file
            rec_atoms: prody.AtomGroup = parse_protein(rec_file)

            # get rdkit molecule from ligand, as well as atom positions/features
            ligand, lig_atom_positions, lig_atom_features = parse_ligand(lig_file, element_map=lig_element_map)

            # get all protein atoms that form the binding pocket
            pocket_atom_positions, pocket_atom_features, pocket_atom_mask \
            = get_pocket_atoms(rec_atoms, 
                lig_atom_positions, 
                box_padding=dataset_config["lig_box_padding"], 
                pocket_cutoff=dataset_config["pocket_cutoff"], 
                element_map=rec_element_map)

            # get boltzmann probabilities for OT loss
            # TODO: implement computing boltzmann probabilities for OT loss
            if dataset_config['use_boltzmann_ot']:
                ot_loss_boltzmann_weights = get_ot_loss_weights()

            # build receptor graph
            receptor_graph = build_receptor_graph(pocket_atom_positions,
                pocket_atom_features, 
                dataset_config['receptor_k'], 
                dataset_config['pocket_edge_algorithm'])

            # define filepaths for saving processed data
            pair_dir = args.output_dir / split_key / str(pair_idx)
            pair_dir.mkdir(exist_ok=True, parents=True)

            # save receptor graph
            receptor_graph_path = pair_dir / f'rec_graph.dgl'
            dgl.save_graphs(str(receptor_graph_path), receptor_graph)

            # save ligand data
            ligand_data_path = pair_dir / f'ligand_data.pt'
            payload = {'lig_atom_positions': lig_atom_positions, 'lig_atom_features': lig_atom_features}
            with open(ligand_data_path, 'wb') as f:
                torch.save(payload, f)
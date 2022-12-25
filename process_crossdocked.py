import argparse
import yaml
import torch
from pathlib import Path
import prody
from typing import Dict
import dgl
import pickle
import numpy as np

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights, center_complex, Unparsable)


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
    rec_elements = dataset_config['rec_elements']
    rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(rec_elements) }
    rec_element_map['other'] = len(rec_elements)

    lig_elements = dataset_config['lig_elements']
    lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(lig_elements) }
    lig_element_map['other'] = len(lig_elements)

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
        data = []
        rec_files = []
        lig_files = []
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

            # note: we used to center the complexes. specifically, we would remove the ligand COM from the 
            # ligand and receptor atom positions. however, i've refactored our training/sampling method
            # such that it does not depend on the assumption that the input structures are in a particular reference frame

            # record this pair
            data.append({
                'receptor_graph': receptor_graph,
                'lig_atom_positions': lig_atom_positions,
                'lig_atom_features': lig_atom_features
            })

            # record receptor and ligand files
            rec_files.append(str(rec_file))
            lig_files.append(str(lig_file))

            # increment dataset_idx
            dataset_idx += 1


        # save data for this split
        data_filepath = output_dir / f'{split_key}.pkl'
        with open(data_filepath, 'wb') as f:
            pickle.dump(data, f)

        filenames = output_dir / f'{split_key}_filenames.pkl'
        with open(filenames, 'wb') as f:
            pickle.dump({'rec_files': rec_files, 'lig_files': lig_files}, f)
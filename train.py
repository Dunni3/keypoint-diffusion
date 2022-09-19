import argparse
import pathlib
import yaml

from data_processing.pdbbind_dataset import PDBbind, get_pdb_dataloader

import dgl
from dgl.dataloading import GraphDataLoader

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/dev_config.yml')
    args = p.parse_args()
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)

    # TODO: allow configs to be specified by commandline arguments

    return config_dict

def main():
    args = parse_arguments()
    dataset = PDBbind(name='train', **args['dataset_config'])

    dataloader = get_pdb_dataloader(dataset, batch_size=2, num_workers=1)

    for rec_graphs, lig_atom_positions, lig_atom_features in dataloader:
        pass


if __name__ == "__main__":
    main()
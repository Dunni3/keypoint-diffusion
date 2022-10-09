import argparse
import pathlib
import yaml

from data_processing.pdbbind_dataset import PDBbind, get_pdb_dataloader
from models.receptor_encoder import ReceptorEncoder

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

    test_rec_graph, _, _ = iter(dataloader).next()
    n_rec_atom_features = test_rec_graph.ndata['h'].shape[1]

    for rec_graphs, lig_atom_positions, lig_atom_features in dataloader:
        break

    rec_encoder = ReceptorEncoder(n_egnn_convs=1, n_keypoints=10, in_n_node_feat=n_rec_atom_features, hidden_n_node_feat=32, out_n_node_feat=32)
    rec_encoder(rec_graphs)
    print('meep!')


if __name__ == "__main__":
    main()
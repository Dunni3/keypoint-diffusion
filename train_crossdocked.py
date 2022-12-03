import argparse
import pathlib
import yaml

from data_processing.crossdocked.dataset import CrossDockedDataset, get_dataloader
from models.dynamics import LigRecDynamics
from models.receptor_encoder import ReceptorEncoder
from models.ligand_diffuser import LigandDiffuser
from losses.rec_encoder_loss import ReceptorEncoderLoss
import torch

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

    dev_dataset_path = args['dataset_location']['train']
    dataset = CrossDockedDataset(name='train', processed_data_dir=dev_dataset_path, **args['dataset_config'])
    rec_encoder_loss_fn = ReceptorEncoderLoss(**args['rec_encoder_loss_config'])

    n_timesteps = 1000

    # TODO: how do we / should we normalize features??
    dataloader = get_dataloader(dataset, batch_size=2, num_workers=1)

    # get number of ligand and receptor atom features
    test_rec_graph, test_lig_pos, test_lig_feat = dataset[0]
    n_rec_atom_features = test_rec_graph.ndata['h_0'].shape[1]
    n_lig_feat = test_lig_feat.shape[1]
    n_kp_feat = args["rec_encoder_config"]["out_n_node_feat"]

    # create receptor encoder
    rec_encoder = ReceptorEncoder(n_egnn_convs=2, 
        n_keypoints=args["rec_encoder_config"]["n_keypoints"], 
        in_n_node_feat=n_rec_atom_features, 
        hidden_n_node_feat=args["rec_encoder_config"]["hidden_n_node_feat"], 
        out_n_node_feat=args["rec_encoder_config"]["out_n_node_feat"])

    # create diffusion model
    model = LigandDiffuser(n_lig_feat, n_kp_feat)

    for rec_graphs, lig_atom_positions, lig_atom_features in dataloader:

        # TODO: add random translations to the complex positions
        
        # run receptor encoder
        kp_pos, kp_feat = rec_encoder(rec_graphs)

        # add noise to ligand and predict noise
        noise_loss = model(lig_atom_positions, lig_atom_features, kp_pos, kp_feat)

        # compute receptor encoder loss
        ot_loss = rec_encoder_loss_fn(kp_pos, rec_graphs)

        loss = noise_loss + ot_loss*args['training_config']['rec_encoder_loss_weight']

    print('meep!')


if __name__ == "__main__":
    main()
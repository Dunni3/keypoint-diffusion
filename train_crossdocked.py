import argparse
import pathlib
import yaml
import pickle
from collections import defaultdict

from data_processing.crossdocked.dataset import CrossDockedDataset, get_dataloader
from models.dynamics import LigRecDynamics
from models.receptor_encoder import ReceptorEncoder
from models.ligand_diffuser import LigandDiffuser
import torch
import numpy as np

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
    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args['training_config']['batch_size']

    # create datasets
    train_dataset_path = args['dataset_location']['train']
    test_dataset_path = args['dataset_location']['test']
    train_dataset = CrossDockedDataset(name='train', processed_data_dir=train_dataset_path, **args['dataset_config'])
    test_dataset = CrossDockedDataset(name='test', processed_data_dir=test_dataset_path, **args['dataset_config'])
    
    n_timesteps = 1000

    # create dataloaders
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=1)

    # get number of ligand and receptor atom features
    test_rec_graph, test_lig_pos, test_lig_feat = train_dataset[0]
    n_rec_atom_features = test_rec_graph.ndata['h_0'].shape[1]
    n_lig_feat = test_lig_feat.shape[1]
    n_kp_feat = args["rec_encoder_config"]["out_n_node_feat"]

    rec_encoder_config = args["rec_encoder_config"]
    rec_encoder_config["in_n_node_feat"] = n_rec_atom_features

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat, 
        rec_encoder_config, 
        args['rec_encoder_loss_config'],
        **args['diffusion_config']
        ).to(device=device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args["training_config"]['learning_rate'],
        weight_decay=args["training_config"]['weight_decay'])

    epoch_losses = defaultdict(list)
    for epoch_idx in range(args['training_config']['epochs']):


        # train for 1 epoch
        batch_losses = []
        batch_l2_losses = []
        batch_ot_losses = []
        for rec_graphs, lig_atom_positions, lig_atom_features in train_dataloader:

            rec_graphs = rec_graphs.to(device)
            lig_atom_positions = [ arr.to(device) for arr in lig_atom_positions ]
            lig_atom_features = [ arr.to(device) for arr in lig_atom_features ]

            optimizer.zero_grad()
            # TODO: add random translations to the complex positions

            # encode receptor, add noise to ligand and predict noise
            noise_loss, ot_loss = model(rec_graphs, lig_atom_positions, lig_atom_features)


            # combine losses
            loss = noise_loss + ot_loss*args['training_config']['rec_encoder_loss_weight']

            batch_losses.append(loss.detach().cpu())
            batch_l2_losses.append(noise_loss.detach().cpu())
            batch_ot_losses.append(ot_loss.detach().cpu())

            loss.backward()
            if args['training_config']['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['training_config']['max_norm'])
            optimizer.step()

        
        mean_loss = np.array(batch_losses).mean()
        mean_l2_loss = np.array(batch_l2_losses).mean()
        mean_ot_loss = np.array(batch_ot_losses).mean()

        epoch_losses['total'].append(mean_loss)
        epoch_losses['l2'].append(mean_l2_loss)
        epoch_losses['ot'].append(mean_ot_loss)

        print(f'Epoch {epoch_idx+1}')
        print(f'Mean Batch Loss = {mean_loss:.2f}')
        print(f'Mean L2 Loss = {mean_l2_loss:.4f}')
        print(f'Mean OT Loss = {mean_ot_loss:.2f}')
        print('\n')


    with open(args['training_config']['metrics_file'], 'wb') as f:
        pickle.dump(epoch_losses, f)


if __name__ == "__main__":
    main()
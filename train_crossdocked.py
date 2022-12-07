import argparse
import yaml
import pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import time
import shutil

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
    p.add_argument('--config', type=str, default='configs/dev_config.yml')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # TODO: allow configs to be specified by commandline arguments

    return args, config_dict


def test_model(model, test_dataloader, args, device):

    l2_losses = []
    losses = []
    ot_losses = []

    for _ in range(args['training_config']['test_epochs']):
        for rec_graphs, lig_atom_positions, lig_atom_features in test_dataloader:

            rec_graphs = rec_graphs.to(device)
            lig_atom_positions = [ arr.to(device) for arr in lig_atom_positions ]
            lig_atom_features = [ arr.to(device) for arr in lig_atom_features ]

            noise_loss, ot_loss = model(rec_graphs, lig_atom_positions, lig_atom_features)

            # combine losses
            loss = noise_loss + ot_loss*args['training_config']['rec_encoder_loss_weight']

            l2_losses.append(noise_loss.detach().cpu())
            losses.append(loss.detach().cpu())
            ot_losses.append(ot_loss.detach().cpu())

    loss_dict = {
        'l2_loss': np.array(l2_losses).mean(),
        'total_loss': np.array(losses).mean(),
        'ot_loss': np.array(ot_losses).mean()
    }
    return loss_dict


def main():
    script_args, args = parse_arguments()
    # torch.autograd.set_detect_anomaly(True)

    # create output directory
    now = datetime.now().strftime('%m%d%H%M%S')
    results_dir = Path(args['experiment']['results_dir'])
    output_dir_name = f"{args['experiment']['name']}_{now}"
    output_dir = results_dir / output_dir_name
    output_dir.mkdir()

    # copy input config file to output directory
    input_config_file = Path(script_args.config).resolve()
    new_configfile_loc = output_dir / input_config_file.name
    shutil.copy(input_config_file, new_configfile_loc)

    # create metrics files and lists to store metrics
    test_metrics_file = output_dir / 'test_metrics.pkl'
    train_metrics_file = output_dir / 'train_metrics.pkl'
    test_metrics = []
    train_metrics = []

    # set random seed
    torch.manual_seed(42)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}', flush=True)

    # get batch size
    batch_size = args['training_config']['batch_size']

    # create datasets
    train_dataset_path = args['dataset_location']['train']
    test_dataset_path = args['dataset_location']['test']
    train_dataset = CrossDockedDataset(name='train', processed_data_dir=train_dataset_path, **args['dataset_config'])
    test_dataset = CrossDockedDataset(name='test', processed_data_dir=test_dataset_path, **args['dataset_config'])

    # create dataloaders
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=args['training_config']['num_workers'])
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=args['training_config']['num_workers'])

    # get number of ligand and receptor atom features
    test_rec_graph, test_lig_pos, test_lig_feat = train_dataset[0]
    n_rec_atom_features = test_rec_graph.ndata['h_0'].shape[1]
    n_lig_feat = test_lig_feat.shape[1]
    n_kp_feat = args["rec_encoder_config"]["out_n_node_feat"]

    print(f'{n_rec_atom_features=}')
    print(f'{n_lig_feat=}', flush=True)

    rec_encoder_config = args["rec_encoder_config"]
    rec_encoder_config["in_n_node_feat"] = n_rec_atom_features
    args["rec_encoder_config"]["in_n_node_feat"] = n_rec_atom_features

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat,
        dynamics_config=args['dynamics'], 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=args['rec_encoder_loss_config']
        ).to(device=device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args["training_config"]['learning_rate'],
        weight_decay=args["training_config"]['weight_decay'])

    # add info that is needed to reconstruct the model later into args
    # this should definitely be done in a cleaner way. and it can be. i just dont have time
    args["reconstruction"] = {
        'n_lig_feat': n_lig_feat,
        'n_rec_atom_feat': n_rec_atom_features
    }

    # save args to output dir
    arg_fp = output_dir / 'args.pkl'
    with open(arg_fp, 'wb') as f:
        pickle.dump(args, f)

    training_start = time.time()
    train_losses = []
    train_l2_losses = []
    train_ot_losses = []
    iterations_per_epoch = len(train_dataset) / batch_size
    for epoch_idx in range(args['training_config']['epochs']):

        
        iter_idx = -1
        for rec_graphs, lig_atom_positions, lig_atom_features in train_dataloader:

            iter_idx += 1

            # TODO: remove this later. I'm just keeping it right now for debugging purposes. There is a bug 
            # where the training loop hangs on the first iteration. Haven't been able to reproduce yet.
            if iter_idx < 50:
                print(f'{iter_idx=}, {time.time() - training_start:.2f} seconds since start', flush=True)

            rec_graphs = rec_graphs.to(device)
            lig_atom_positions = [ arr.to(device) for arr in lig_atom_positions ]
            lig_atom_features = [ arr.to(device) for arr in lig_atom_features ]

            optimizer.zero_grad()
            # TODO: add random translations to the complex positions

            # encode receptor, add noise to ligand and predict noise
            noise_loss, ot_loss = model(rec_graphs, lig_atom_positions, lig_atom_features)


            # combine losses
            loss = noise_loss + ot_loss*args['training_config']['rec_encoder_loss_weight']

            # record losses for this batch
            train_losses.append(loss.detach().cpu())
            train_l2_losses.append(noise_loss.detach().cpu())
            train_ot_losses.append(ot_loss.detach().cpu())

            loss.backward()
            if args['training_config']['clip_grad']:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args['training_config']['clip_value'])
            optimizer.step()

            # save the model if necessary
            if iter_idx % args['training_config']['save_interval'] == 0 and iter_idx > 0:
                file_name = f'model_epoch_{epoch_idx}_iter_{iter_idx}.pt'
                file_path = output_dir / file_name
                most_recent_model = output_dir / 'model.pt'
                torch.save(model.state_dict(), str(file_path))
                torch.save(model.state_dict(), str(most_recent_model))

            # test the model if necessary
            if iter_idx % args['training_config']['test_interval'] == 0:
                model.eval()
                test_metrics_row = test_model(model, test_dataloader, args, device=device)
                model.train()

                test_metrics_row['epoch'] = epoch_idx
                test_metrics_row['iter'] = iter_idx
                test_metrics_row['time_passed'] = time.time() - training_start 
                test_metrics.append(test_metrics_row)
                with open(test_metrics_file, 'wb') as f:
                    pickle.dump(test_metrics, f)

                print('test metrics')
                print(*[ f'{k} = {v:.2f}' for k,v in test_metrics_row.items()], sep='\n', flush=True)
                print('\n')


            # record train metrics if necessary
            if iter_idx % args['training_config']['train_metrics_interval'] == 0:

                train_metrics_row = {
                    'total_loss': np.array(train_losses).mean(),
                    'l2_loss': np.array(train_l2_losses).mean(),
                    'ot_loss': np.array(train_ot_losses).mean(),
                    'epoch': epoch_idx,
                    'iter': iter_idx,
                    'time_passed': time.time() - training_start
                }
                train_metrics.append(train_metrics_row)
                with open(train_metrics_file, 'wb') as f:
                    pickle.dump(train_metrics, f)

                print('training metrics')
                print(*[ f'{k} = {v:.2f}' for k,v in train_metrics_row.items()], sep='\n', flush=True)
                print('\n')

                train_losses, train_l2_losses, train_ot_losses, = [], [], []

    
    # after exiting the training loop, save the final model file
    most_recent_model = output_dir / 'model.pt'
    torch.save(model.state_dict(), str(most_recent_model))


if __name__ == "__main__":
    main()
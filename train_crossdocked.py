import argparse
import sys
import yaml
import pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import time
import shutil
import wandb

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

    # rec_encoder_group = p.add_argument_group('receptor encoder')
    # rec_encoder_group.add_argument('--n_keypoints', type=int, default=20, help="number of keypoints produced by receptor encoder module")
    # rec_encoder_group.add_argument('--n_convs_encoder', type=int, default=6, help="number of graph convolutions in receptor encoder")
    # rec_encoder_group.add_argument('--encoder_hidden_feats', type=int, default=256, help="number of hidden features in receptor encoder")
    # rec_encoder_group.add_argument('--keypoint_feats', type=int, default=256, help='number of features for receptor keypoints')
    
    # dynamics_group = p.add_argument_group('dynamics')
    # dynamics_group.add_argument('--n_convs_dynamics', type=int, default=6, help='number of graph convolutions in the dynamics model')
    # dynamics_group.add_argument('--keypoint_k', type=int, default=6, help='K for keypoint -> ligand KNN graph')
    # dynamics_group.add_argument('--ligand_k', type=int, default=8, help='K for ligand -> ligand KNN graph')
    # dynamics_group.add_argument('--use_tanh', type=bool, default=True, help='whether to place tanh activation on coordinate MLP output')

    # training_group = p.add_argument_group('training')
    # training_group.add_argument('--rec_encoder_loss_weight', type=float, default=0.1, help='relative weight applied to receptor encoder OT loss')
    # training_group.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    # training_group.add_argument('--weight_decay', type=float, default=1e-9)
    # training_group.add_argument('--clip_grad', type=bool, default=True, help='whether to clip gradients')
    # training_group.add_argument('--clip_value', type=float, default=1.5, help='max gradient value for clipping')
    # training_group.add_argument('--epochs', type=int, default=1000)
    # training_group.add_argument('--batch_size', type=int, default=32)
    # training_group.add_argument('--test_interval', type=float, default=1, help="evaluate on test set every test_interval epochs")
    # training_group.add_argument('--train_metrics_interval', type=float, default=1, help="report training metrics every train_metrics_interval epochs")
    # training_group.add_argument('--test_epochs', type=float, default=2, help='number of epochs to run on test set evaluation')
    # training_group.add_argument('--num_workers', type=int, default=1, help='num_workers argument for pytorch dataloader')
    # TODO: how do i merge commandline arguments with config file arguments?

    p.add_argument('--config', type=str, default='configs/dev_config.yml')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    return args, config_dict

@torch.no_grad()
def test_model(model, test_dataloader, args, device):

    l2_losses = []
    losses = []
    ot_losses = []

    for _ in range(args['training']['test_epochs']):
        for rec_graphs, lig_atom_positions, lig_atom_features in test_dataloader:

            rec_graphs = rec_graphs.to(device)
            lig_atom_positions = [ arr.to(device) for arr in lig_atom_positions ]
            lig_atom_features = [ arr.to(device) for arr in lig_atom_features ]

            noise_loss, ot_loss = model(rec_graphs, lig_atom_positions, lig_atom_features)

            # combine losses
            loss = noise_loss + ot_loss*args['training']['rec_encoder_loss_weight']

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
    new_configfile_loc = output_dir / 'config.yml'
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
    batch_size = args['training']['batch_size']

    # create datasets
    dataset_path = Path(args['dataset']['location']) 
    train_dataset_path = str(dataset_path / 'train.pkl') 
    test_dataset_path = str(dataset_path / 'test.pkl')
    train_dataset = CrossDockedDataset(name='train', processed_data_file=train_dataset_path, **args['dataset'])
    test_dataset = CrossDockedDataset(name='test', processed_data_file=test_dataset_path, **args['dataset'])

    # compute number of iterations per epoch - necessary for deciding when to do test evaluations/saves/etc. 
    iterations_per_epoch = len(train_dataset) / batch_size

    # create dataloaders
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=args['training']['num_workers'], shuffle=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=args['training']['num_workers'])

    # get number of ligand and receptor atom features
    test_rec_graph, test_lig_pos, test_lig_feat = train_dataset[0]
    n_rec_atom_features = test_rec_graph.ndata['h_0'].shape[1]
    n_lig_feat = test_lig_feat.shape[1]
    n_kp_feat = args["rec_encoder"]["out_n_node_feat"]

    print(f'{n_rec_atom_features=}')
    print(f'{n_lig_feat=}', flush=True)

    rec_encoder_config = args["rec_encoder"]
    rec_encoder_config["in_n_node_feat"] = n_rec_atom_features
    args["rec_encoder"]["in_n_node_feat"] = n_rec_atom_features

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat,
        n_timesteps=args['diffusion']['n_timesteps'],
        dynamics_config=args['dynamics'], 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=args['rec_encoder_loss']
        ).to(device=device)

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args["training"]['learning_rate'],
        weight_decay=args["training"]['weight_decay'])

    # check if we are using cylic learning rates
    use_cyclic_lr = args['training']['cyclic_lr']['use_cyclic_lr']

    # create learning rate schedulers
    if use_cyclic_lr:
        cyclic_args = args['training']['cyclic_lr']
        cyclic_lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, 
            base_lr=cyclic_args['base_lr'], 
            max_lr=cyclic_args['max_lr'],
            step_size_up=int(iterations_per_epoch*cyclic_args['step_size_up_frac']),
            cycle_momentum=False)

    # initialize wandb
    wandb_init_kwargs = args['wandb']['init_kwargs']
    wandb_init_kwargs['name'] = args['experiment']['name']
    wandb.init(config=args, **wandb_init_kwargs)

    # watch model if desired
    if args['wandb']['watch_model']:
        wandb.watch(model, **args['wandb']['watch_kwargs'])

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
    
    # create empty lists to record per-batch losses
    train_losses = []
    train_l2_losses = []
    train_ot_losses = []
    

    # create markers for deciding when to evaluate on the test set, report training metrics, save the model
    test_report_marker = 0 # measured in epochs
    train_report_marker = 0 # measured in epochs
    save_marker = 0

    # record start time for training
    training_start = time.time()

    model.train()
    for epoch_idx in range(args['training']['epochs']):

        for iter_idx, iter_data in enumerate(train_dataloader):
            rec_graphs, lig_atom_positions, lig_atom_features = iter_data

            current_epoch = epoch_idx + iter_idx/iterations_per_epoch

            # TODO: remove this later. I'm just keeping it right now for debugging purposes. There is a bug 
            # where the training loop hangs on the first iteration. Haven't been able to reproduce yet.
            if iter_idx < 25:
                print(f'{iter_idx=}, {time.time() - training_start:.2f} seconds since start', flush=True)

            rec_graphs = rec_graphs.to(device)
            lig_atom_positions = [ arr.to(device) for arr in lig_atom_positions ]
            lig_atom_features = [ arr.to(device) for arr in lig_atom_features ]

            optimizer.zero_grad()
            # TODO: add random translations to the complex positions

            # encode receptor, add noise to ligand and predict noise
            noise_loss, ot_loss = model(rec_graphs, lig_atom_positions, lig_atom_features)

            # combine losses
            loss = noise_loss + ot_loss*args['training']['rec_encoder_loss_weight']

            # record losses for this batch
            train_losses.append(loss.detach().cpu())
            train_l2_losses.append(noise_loss.detach().cpu())
            train_ot_losses.append(ot_loss.detach().cpu())

            loss.backward()
            if args['training']['clip_grad']:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args['training']['clip_value'])
            optimizer.step()

            # save the model if necessary
            if current_epoch - save_marker >= args['training']['save_interval']:
            # if iter_idx % args['training']['save_interval'] == 0 and iter_idx > 0:
                save_marker = current_epoch # update save marker
                file_name = f'model_epoch_{epoch_idx}_iter_{iter_idx}.pt' # where to save current model
                file_path = output_dir / file_name 
                most_recent_model = output_dir / 'model.pt' # filepath of most recently saved model - note this is always the same path
                torch.save(model.state_dict(), str(file_path)) # save current model
                torch.save(model.state_dict(), str(most_recent_model)) # save most recent model

            # test the model if necessary
            if current_epoch - test_report_marker >= args['training']['test_interval'] or current_epoch == 0:
                test_report_marker = current_epoch

                model.eval()
                test_metrics_row = test_model(model, test_dataloader, args, device=device)
                model.train()

                test_metrics_row['epoch_exact'] = current_epoch
                test_metrics_row['epoch'] = epoch_idx
                test_metrics_row['iter'] = iter_idx
                test_metrics_row['time_passed'] = time.time() - training_start 
                test_metrics.append(test_metrics_row)
                with open(test_metrics_file, 'wb') as f:
                    pickle.dump(test_metrics, f)

                print('test metrics')
                print(*[ f'{k} = {v:.2f}' for k,v in test_metrics_row.items()], sep='\n', flush=True)
                print('\n')

                # log test metrics to wandb
                test_metrics_wandb = test_metrics_row.copy()
                for key in list(test_metrics_wandb): # prepend 'train' onto all loss metrics
                    if 'loss' in key:
                        new_key = f'test_{key}'
                        test_metrics_wandb[new_key] = test_metrics_wandb[key]
                        del test_metrics_wandb[key]
                wandb.log(test_metrics_wandb)


            # record train metrics if necessary
            if current_epoch - train_report_marker >= args['training']['train_metrics_interval']:
                train_report_marker = current_epoch


                train_metrics_row = {
                    'total_loss': np.array(train_losses).mean(),
                    'l2_loss': np.array(train_l2_losses).mean(),
                    'ot_loss': np.array(train_ot_losses).mean(),
                    'epoch': epoch_idx,
                    'epoch_exact': current_epoch,
                    'iter': iter_idx,
                    'time_passed': time.time() - training_start
                }
                train_metrics.append(train_metrics_row)
                with open(train_metrics_file, 'wb') as f:
                    pickle.dump(train_metrics, f)

                print('training metrics')
                print(*[ f'{k} = {v:.2f}' for k,v in train_metrics_row.items()], sep='\n', flush=True)
                print('\n')

                # log train metrics to wandb
                train_metrics_wandb = train_metrics_row.copy()
                for key in list(train_metrics_wandb): # prepend 'train' onto all loss metrics
                    if 'loss' in key:
                        new_key = f'train_{key}'
                        train_metrics_wandb[new_key] = train_metrics_wandb[key]
                        del train_metrics_wandb[key]
                wandb.log(train_metrics_wandb)

                # TODO: remove this line, for debugging only!!
                # if max(train_losses) > 1e4:
                #     sys.exit()

                train_losses, train_l2_losses, train_ot_losses, = [], [], []
                

            # apply cyclic LR update if necessary
            if use_cyclic_lr:
                cyclic_lr_sched.step()

    # after exiting the training loop, save the final model file
    most_recent_model = output_dir / 'model.pt'
    torch.save(model.state_dict(), str(most_recent_model))


if __name__ == "__main__":
    main()
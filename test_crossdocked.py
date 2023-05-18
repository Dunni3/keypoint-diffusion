import argparse
import time
import yaml
from pathlib import Path
import torch
import numpy as np
import prody
from rdkit import Chem
import shutil
import pickle
from tqdm import trange

from data_processing.crossdocked.dataset import CrossDockedDataset
from data_processing.make_bindingmoad_pocketfile import write_pocket_file
from models.ligand_diffuser import LigandDiffuser
from utils import write_xyz_file
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties
from analysis.pocket_minimization import pocket_minimization

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, default=None, help='directory of training result for the model')
    p.add_argument('--model_file', type=str, default=None, help='Path to file containing model weights. If not specified, the most recently saved weights file in model_dir will be used')
    p.add_argument('--samples_per_pocket', type=int, default=100)
    p.add_argument('--avg_validity', type=float, default=1, help='average fraction of generated molecules which are valid')
    p.add_argument('--max_batch_size', type=int, default=128, help='maximum feasible batch size due to memory constraints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='test_results/')
    p.add_argument('--max_tries', type=int, default=3, help='maximum number of batches to sample per pocket')
    p.add_argument('--dataset_size', type=int, default=None, help='truncate test dataset, for debugging only')
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--dataset', type=str, default='bindingmoad')
    p.add_argument('--dataset_idx', type=int, default=None)

    # p.add_argument('--no_metrics', action='store_true')
    # p.add_argument('--no_minimization', action='store_true')
    p.add_argument('--ligand_only_minimization', action='store_true')
    p.add_argument('--pocket_minimization', action='store_true')
    
    args = p.parse_args()

    if args.model_file is not None and args.model_dir is not None:
        raise ValueError('only model_file or model_dir can be specified but not both')
    
    if args.dataset not in ['crossdocked', 'bindingmoad']:
        raise ValueError('unsupported dataset')

    return args

def make_reference_files(dataset_idx: int, dataset: CrossDockedDataset, output_dir: Path) -> Path:

    # get original receptor and ligand files
    ref_rec_file, ref_lig_file = dataset.get_files(dataset_idx)
    ref_rec_file = Path(ref_rec_file)
    ref_lig_file = Path(ref_lig_file)

    # # get receptor object and atom coordinates
    # rec: prody.AtomGroup = prody.parsePDB(str(ref_rec_file))
    # rec_coords = rec.getCoords()

    # # get ligand object 
    # suppl = Chem.SDMolSupplier(str(ref_lig_file), sanitize=False, removeHs=remove_hydrogen)
    # ligands = list(suppl)
    # if len(ligands) > 1:
    #     raise NotImplementedError('Multiple ligands found. Code is not written to handle multiple ligands.')
    # ligand = ligands[0]

    # # get atom positions
    # ligand_conformer = ligand.GetConformer()
    # ligand_atom_positions = ligand_conformer.GetPositions()

    # ligand_com = ligand_atom_positions.mean(axis=0, keepdims=True)

    # # remove ligand COM from receptor coordinates
    # rec.setCoords( rec_coords - ligand_com )

    # # remove ligand COM from ligand coordinates
    # new_lig_pos = ligand_atom_positions - ligand_com
    # for i in range(ligand.GetNumAtoms()):
    #     x,y,z = new_lig_pos[i]
    #     ligand_conformer.SetAtomPosition(i,Point3D(x,y,z))

    # get filepath of new ligand and receptor files
    centered_lig_file = output_dir / ref_lig_file.name
    centered_rec_file = output_dir / ref_rec_file.name

    # # write ligand to new file
    # lig_writer = Chem.SDWriter(str(centered_lig_file))
    # lig_writer.write(ligand)
    # lig_writer.close()

    # # write receptor to new file
    # prody.writePDB(str(centered_rec_file), rec)

    shutil.copy(ref_rec_file, centered_rec_file)
    shutil.copy(ref_lig_file, centered_lig_file)

    return output_dir

def write_ligands(mols, filepath: Path):
    writer = Chem.SDWriter(str(filepath))

    for mol in mols:
        writer.write(mol)

    writer.close()


def main():
    
    cmd_args = parse_arguments()

    # get output dir path and create the directory
    output_dir = Path(cmd_args.output_dir)
    output_dir.mkdir(exist_ok=True)
    mols_dir = output_dir / 'sampled_mols'
    mols_dir.mkdir(exist_ok=True)

    # get filepath of config file within model_dir
    if cmd_args.model_dir is not None:
        model_dir = Path(cmd_args.model_dir)
        model_file = model_dir / 'model.pt'
    elif cmd_args.model_file is not None:
        model_file = Path(cmd_args.model_file)
        model_dir = model_file.parent
    
    # get config file
    config_file = model_dir / 'config.yml'

    # load model configuration
    with open(config_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}', flush=True)

    # set random seeds
    torch.manual_seed(cmd_args.seed)

    # create test dataset object
    dataset_path = Path(args['dataset']['location']) 
    test_dataset_path = str(dataset_path / f'{cmd_args.split}.pkl')
    test_dataset = CrossDockedDataset(name=cmd_args.split, processed_data_file=test_dataset_path, **args['dataset'])

    # get number of ligand and receptor atom features
    n_lig_feat = args['reconstruction']['n_lig_feat']
    n_kp_feat = args["rec_encoder"]["out_n_node_feat"]

    rec_encoder_config = args["rec_encoder"]

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat,
        processed_dataset_dir=Path(args['dataset']['location']),
        dynamics_config=args['dynamics'], 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=args['rec_encoder_loss'],
        **args['diffusion']).to(device=device)

    # load model weights
    model.load_state_dict(torch.load(model_file))
    model.eval()


    pocket_mols = []
    pocket_sampling_times = []
    keypoints = []

    # generate the iterator over the dataset
    if cmd_args.dataset_idx is None:
        # truncate the dataset if we need to
        if cmd_args.dataset_size is not None:
            dataset_size = cmd_args.dataset_size
        else:
            dataset_size = len(test_dataset)
        dataset_iterator = trange(dataset_size)
    else:
        dataset_iterator = trange(cmd_args.dataset_idx, cmd_args.dataset_idx+1)

    # iterate over dataset and draw samples for each pocket
    for dataset_idx in dataset_iterator:

        pocket_sample_start = time.time()

        # get receptor graph and reference ligand positions/features from test set
        rec_graph, ref_lig_pos, ref_lig_feat = test_dataset[dataset_idx]
        ref_rec_file, ref_lig_file = test_dataset.get_files(dataset_idx) # get original rec/lig files

        # move data to gpu
        rec_graph = rec_graph.to(device)
        # ref_lig_pos.to(device)
        # ref_lig_feat.to(device)
        
        n_lig_atoms: int = ref_lig_pos.shape[0] # get number of atoms in the ligand
        atoms_per_ligand = torch.ones(size=(cmd_args.max_batch_size,), dtype=int, device=device)*n_lig_atoms # input to sampling function 

        # encode the receptor
        kp_pos, kp_feat, init_rec_atom_com, init_kp_com = model.encode_receptors([rec_graph])

        # create batch_size copies of the encoded receptor
        kp_pos = [ kp_pos[0].detach().clone() for _ in range(cmd_args.max_batch_size) ]
        kp_feat = [ kp_feat[0].detach().clone() for _ in range(cmd_args.max_batch_size) ]
        init_rec_atom_com = [ init_rec_atom_com[0].detach().clone() for _ in range(cmd_args.max_batch_size) ]
        init_kp_com = [ init_kp_com[0].detach().clone() for _ in range(cmd_args.max_batch_size) ]

        pocket_raw_mols = []

        for attempt_idx in range(cmd_args.max_tries):

            n_mols_needed = cmd_args.samples_per_pocket - len(pocket_raw_mols)
            n_mols_to_generate = int( n_mols_needed / (cmd_args.avg_validity*0.95) ) + 1
            batch_size = min(n_mols_to_generate, cmd_args.max_batch_size)

            # sample ligand atom positions/features
            batch_lig_pos, batch_lig_feat = model.sample_from_encoded_receptors(
                kp_pos[:batch_size], 
                kp_feat[:batch_size], 
                init_rec_atom_com[:batch_size], 
                init_kp_com[:batch_size], 
                atoms_per_ligand[:batch_size])

            # convert positions/features to rdkit molecules
            for lig_idx in range(batch_size):

                # convert lig atom features to atom elements
                element_idxs = torch.argmax(batch_lig_feat[lig_idx], dim=1).tolist()
                atom_elements = test_dataset.lig_atom_idx_to_element(element_idxs)

                # build molecule
                mol = build_molecule(batch_lig_pos[lig_idx], atom_elements, add_hydrogens=False, sanitize=True, largest_frag=True, relax_iter=0)

                if mol is not None:
                    pocket_raw_mols.append(mol)

            # stop generating molecules if we've made enough
            if len(pocket_raw_mols) >= cmd_args.samples_per_pocket:
                break

        pocket_sample_time = time.time() - pocket_sample_start
        pocket_sampling_times.append(pocket_sample_time)

        # create directory for sampled molecules
        pocket_dir = mols_dir / f'pocket_{dataset_idx}'
        pocket_dir.mkdir(exist_ok=True)

        # save pocket sample time
        with open(pocket_dir / 'sample_time.txt', 'w') as f:
            f.write(f'{pocket_sample_time:.2f}')
        with open(pocket_dir / 'sample_time.pkl', 'wb') as f:
            pickle.dump(pocket_sample_time, f)


        # write receptor to the pocket dir
        pocket_file = pocket_dir / 'pocket.pdb'
        if args['dataset'] == 'bindingmoad':
            write_pocket_file(ref_rec_file, ref_lig_file, pocket_file, cutoff=args['dataset']['pocket_cutoff'])
            full_rec_file = pocket_dir / 'receptor.pdb'
            shutil.copy(ref_rec_file, full_rec_file)
        elif args['dataset'] == 'crossdocked':
            shutil.copy(ref_rec_file, pocket_file)

        # write the reference files to the pocket dir
        ref_files_dir = pocket_dir / 'reference_files'
        ref_files_dir.mkdir(exist_ok=True)
        shutil.copy(ref_lig_file, ref_files_dir)
        shutil.copy(ref_rec_file, ref_files_dir)

        # give molecules a name
        for idx, mol in enumerate(pocket_raw_mols):
            mol.SetProp('_Name', f'lig_idx_{idx}')

        # ligand-only minimization
        if cmd_args.ligand_only_minimization:
            pocket_lomin_mols = []
            for raw_mol in pocket_raw_mols:
                minimized_mol = process_molecule(Chem.Mol(raw_mol), add_hydrogens=True, relax_iter=200)
                if minimized_mol is not None:
                    pocket_lomin_mols.append(minimized_mol)
            # TODO: write minimized ligands
            ligands_file = pocket_dir / 'minimized_ligands.sdf'
            write_ligands(pocket_lomin_mols, ligands_file)

        # pocket-only minimization
        if cmd_args.pocket_minimization:
            input_mols = [ Chem.Mol(raw_mol) for raw_mol in pocket_raw_mols ]
            pocket_pmin_mols, rmsd_df = pocket_minimization(pocket_file, input_mols, add_hs=True)
            ligands_file = pocket_dir / 'pocket_minimized_ligands.sdf'
            write_ligands(pocket_pmin_mols, ligands_file)
            rmsds_file = pocket_dir / 'pocket_min_rmsds.csv'
            rmsd_df.to_csv(rmsds_file, index=False)


        # remove KP COM, add back in init_kp_com, then save keypoint positions
        keypoint_positions = kp_pos[0]
        keypoint_positions = keypoint_positions - keypoint_positions.mean(dim=0, keepdims=True) + init_kp_com[0]
        
        # write keypoints to an xyz file
        kp_file = pocket_dir / 'keypoints.xyz'
        kp_elements = ['C' for _ in range(keypoint_positions.shape[0]) ]
        write_xyz_file(keypoint_positions, kp_elements, kp_file)


    # compute metrics on the sampled molecules
    # if not cmd_args.no_metrics:
    #     mol_metrics = MoleculeProperties()
    #     all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
    #         mol_metrics.evaluate(pocket_mols)


    #     # save computed metrics
    #     metrics = {
    #         'qed': all_qed, 'sa': all_sa, 'logp': all_logp, 'lipinski': all_lipinski, 'diversity': per_pocket_diversity,
    #         'pocket_sampling_time': pocket_sampling_times
    #     }

    #     metrics_file = output_dir / 'metrics.pkl'
    #     with open(metrics_file, 'wb') as f:
    #         pickle.dump(metrics, f)

    # # save all the sampled molecules, reference files, and keypoints
    # mols_dir = output_dir / 'sampled_mols'
    # mols_dir.mkdir(exist_ok=True)
    # for i, pocket_raw_mols in enumerate(pocket_mols):
    #     pocket_dir = mols_dir / f'pocket_{i}'
    #     pocket_dir.mkdir(exist_ok=True)
    #     pocket_ligands_file = pocket_dir / f'pocket_{i}_ligands.sdf'
    #     write_ligands(pocket_raw_mols, pocket_ligands_file) # write ligands
    #     make_reference_files(i, test_dataset, pocket_dir) # write receptor and reference ligand
        
    #     # write keypoints to an xyz file
    #     kp_file = pocket_dir / 'keypoints.xyz'
    #     kpi = keypoints[i]
    #     kp_elements = ['C' for _ in range(kpi.shape[0]) ]
    #     write_xyz_file(kpi, kp_elements, kp_file)

    # create a summary file
    # if not cmd_args.no_metrics:
    #     summary_file = output_dir / 'summary.txt'
    #     summary_file_contents = ''
    #     for metric_name in metrics.keys():
    #         metric = metrics[metric_name]
    #         if metric_name in ['diversity', 'pocket_sampling_time']:
    #             metric_flattened = metric
    #         else:
    #             metric_flattened = [x for px in metric for x in px]
    #         metric_mean = np.mean(metric_flattened)
    #         metric_std = np.std(metric_flattened)
    #         summary_file_contents += f'{metric_name} = {metric_mean:.3f} \pm {metric_std:.2f}\n'
    #     with open(summary_file, 'w') as f:
    #         f.write(summary_file_contents)

if __name__ == "__main__":

    with torch.no_grad():
        main()
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

from data_processing.crossdocked.dataset import CrossDockedDataset
from models.ligand_diffuser import LigandDiffuser
from utils import write_xyz_file
from analysis.molecule_builder import build_molecule
from analysis.metrics import MoleculeProperties

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True, help='directory of training result for the model')
    p.add_argument('--model_file', type=str, default=None, help='Path to file containing model weights. If not specified, the most recently saved weights file in model_dir will be used')
    p.add_argument('--samples_per_pocket', type=int, default=100)
    p.add_argument('--avg_validity', type=float, default=1, help='average fraction of generated molecules which are valid')
    p.add_argument('--max_batch_size', type=int, deafult=128, help='maximum feasible batch size due to memory constraints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='test_results/')
    p.add_argument('--max_tries', type=int, default=3, help='maximum number of batches to sample per pocket')
    
    args = p.parse_args()

    return args

def make_reference_files(dataset_idx: int, dataset: CrossDockedDataset, output_dir: Path, remove_hydrogen: bool) -> Path:

    output_dir = output_dir / str(dataset_idx)
    output_dir.mkdir(exist_ok=True)

    # get original receptor and ligand files
    ref_rec_file = Path(dataset.filenames['rec_files'][dataset_idx])
    ref_lig_file = Path(dataset.filenames['lig_files'][dataset_idx])

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

    # get filepath of config file within model_dir
    model_dir = Path(cmd_args.model_dir)
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
    test_dataset_path = str(dataset_path / 'test.pkl')
    test_dataset = CrossDockedDataset(name='test', processed_data_file=test_dataset_path, **args['dataset'])

    # get number of ligand and receptor atom features
    n_lig_feat = args['reconstruction']['n_lig_feat']
    n_kp_feat = args["rec_encoder"]["out_n_node_feat"]

    rec_encoder_config = args["rec_encoder"]

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat,
        n_timesteps=args['diffusion']['n_timesteps'],
        keypoint_centered=args['diffusion']['keypoint_centered'],
        dynamics_config=args['dynamics'], 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=args['rec_encoder_loss']
        ).to(device=device)

    # get file for model weights
    if cmd_args.model_file is None:
        model_weights_file = model_dir / 'model.pt'
    else:
        model_weights_file = cmd_args.model_file

    # load model weights
    model.load_state_dict(torch.load(model_weights_file))
    model.eval()


    pocket_mols = []
    pocket_sampling_times = []
    for dataset_idx in len(test_dataset):

        pocket_sample_start = time.time()

        # get receptor graph and reference ligand positions/features from test set
        rec_graph, ref_lig_pos, ref_lig_feat = test_dataset[dataset_idx]
        rec_file_path, ref_lig_file_path = test_dataset.get_files(dataset_idx) # get original rec/lig files

        # move data to gpu
        rec_graph.to(device)
        # ref_lig_pos.to(device)
        # ref_lig_feat.to(device)
        
        n_lig_atoms: int = ref_lig_pos.shape[0] # get number of atoms in the ligand
        atoms_per_ligand = torch.ones(size=(args.max_batch_size,), dtype=int, device=device)*n_lig_atoms # input to sampling function 

        # encode the receptor
        kp_pos, kp_feat, init_rec_atom_com, init_kp_com = model.encode_receptors([rec_graph])

        # create batch_size copies of the encoded receptor
        kp_pos = [ kp_pos[0].detach().clone() for _ in range(args.max_batch_size) ]
        kp_feat = [ kp_feat[0].detach().clone() for _ in range(args.max_batch_size) ]
        init_rec_atom_com = [ init_rec_atom_com[0].detach().clone() for _ in range(args.max_batch_size) ]
        init_kp_com = [ init_kp_com[0].detach().clone() for _ in range(args.max_batch_size) ]

        mols = []

        for attempt_idx in range(args.max_tries):

            n_mols_needed = args.samples_per_pocket - len(mols)
            n_mols_to_generate = int( n_mols_needed / (args.avg_validity*0.95) ) + 1
            batch_size = min(n_mols_to_generate, args.max_batch_size)

            # sample ligand atom positions/features
            batch_lig_pos, batch_lig_feat = model.sample_from_encoded_receptors(
                kp_pos[:batch_size], 
                kp_feat[:batch_size], 
                init_rec_atom_com[:batch_size], 
                init_kp_com[:batch_size], 
                atoms_per_ligand[:batch_size])

            # convert positions/features to rdkit molecules
            for lig_idx in range(args.batch_size):

                # convert lig atom features to atom elements
                element_idxs = torch.argmax(batch_lig_feat[lig_idx], dim=1).tolist()
                atom_elements = test_dataset.lig_atom_idx_to_element(element_idxs)

                # build molecule
                mol = build_molecule(batch_lig_pos[lig_idx], atom_elements, add_hydrogens=True, sanitize=True, largest_frag=True, relax_iter=200)

                if mol is not None:
                    mols.append(mol)


            # stop generating molecules if we've made enough
            if len(mols) >= args.samples_per_pocket:
                break

        pocket_sample_time = time.time() - pocket_sample_start
        pocket_sampling_times.append(pocket_sample_time)
        pocket_mols.append(mols)
        

    # compute metrics on the sampled molecules
    mol_metrics = MoleculeProperties()
    all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
        mol_metrics.evaluate(pocket_mols)


    # save computed metrics
    metrics = {
        'qed': all_qed, 'sa': all_sa, 'logp': all_logp, 'lipinski': all_lipinski, 'diversity': per_pocket_diversity
    }

    metrics_file = output_dir / 'metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    # save all the sampled molecules
    mols_dir = output_dir / 'sampled_mols'
    for i, mols in enumerate(pocket_mols):
        pocket_ligands_file = mols_dir / f'pocket_{i}_ligands.sdf'
        write_ligands(mols, pocket_ligands_file)


if __name__ == "__main__":

    with torch.no_grad():
        main()
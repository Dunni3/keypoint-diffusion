import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import prody
from rdkit import Chem
import shutil

from data_processing.crossdocked.dataset import CrossDockedDataset
from models.ligand_diffuser import LigandDiffuser
from utils import write_xyz_file, make_mol_openbabel

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=True, help='directory of training result for the model')
    p.add_argument('--model_file', type=str, default=None, help='Path to file containing model weights. If not specified, the most recently saved weights file in model_dir will be used')
    p.add_argument('--epoch_sample', type=float, default=None, help='The epoch value of the model checkpoint that you want to sample. If specified, the checkpoint having the closest epoch value to the argument will be used for sampling.')
    p.add_argument('--n_replicates', type=int, default=1)
    p.add_argument('--n_complexes', type=int, default=1)
    # p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--random', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='sampled_mols/')
    

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

def write_sampled_ligands(lig_pos, lig_feat, output_dir: Path, dataset: CrossDockedDataset):

    lig_pos = [ arr.detach().cpu() for arr in lig_pos ]
    lig_feat = [ arr.detach().cpu() for arr in lig_feat ]

    sdf_file = output_dir / 'sampled_mols.sdf'
    writer = Chem.SDWriter(str(sdf_file))

    for lig_idx in range(len(lig_pos)):
        element_idxs = torch.argmax(lig_feat[lig_idx], dim=1).tolist()

        atom_elements = dataset.lig_atom_idx_to_element(element_idxs)

        mol = make_mol_openbabel(lig_pos[lig_idx], atom_elements)

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

    # set random seeds / create rando number generator
    torch.manual_seed(cmd_args.seed)
    rng = np.random.default_rng(42)

    # create test dataset
    dataset_path = Path(args['dataset']['location']) 
    test_dataset_path = str(dataset_path / 'test.pkl')
    test_dataset = CrossDockedDataset(name='test', processed_data_file=test_dataset_path, **args['dataset'])

    # get number of ligand and receptor atom features
    test_rec_graph, test_lig_pos, test_lig_feat = test_dataset[0]
    n_rec_atom_features = test_rec_graph.ndata['h_0'].shape[1]
    n_lig_feat = test_lig_feat.shape[1]
    n_kp_feat = args["rec_encoder"]["out_n_node_feat"]

    rec_encoder_config = args["rec_encoder"]
    rec_encoder_config["in_n_node_feat"] = n_rec_atom_features
    args["rec_encoder"]["in_n_node_feat"] = n_rec_atom_features

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
    
    # get dataset indexes for complexes we are going to sample
    if cmd_args.random:
        dataset_idxs = rng.choice(len(test_dataset), size=cmd_args.n_complexes, replace=False)
    else:
        dataset_idxs = np.arange(cmd_args.n_complexes)

    ref_complex_idx = []
    complex_output_dirs = {}
    for idx in dataset_idxs:

        # create files with the reference ligand/receptor, center them appropriately so that their coordinates
        # align with the output of the diffusion model -> these steps are done by make_reference_files
        # the function make_reference_files then returns the location where the reference files have been written
        complex_output_dir = make_reference_files(idx, test_dataset, output_dir, remove_hydrogen=args['dataset']['remove_hydrogen'])
        complex_output_dirs[idx] = complex_output_dir

        # get the data for the reference complex
        rec_graph, ref_lig_pos, ref_lig_feat = test_dataset[idx]

        # move data to correct device
        rec_graph = rec_graph.to(device)

        # get array specifying the number of nodes in each ligand we sample
        n_nodes = torch.ones(size=(cmd_args.n_replicates,), dtype=int, device=device)*ref_lig_pos.shape[0]

        # get receptor keypoints
        # note the diffusion model does receptor encoding internally,
        # so for sampling this is not strictly necessary, but i would like to visualize the position of the keypoints
        kp_pos, kp_feat = model.rec_encoder(rec_graph)
        kp_pos, kp_feat = kp_pos[0], kp_feat[0] # the keypoints and features are returned as lists of length batch_size, but now our batch size is just 1

        # sample ligands
        lig_pos, lig_feat = model.sample_given_pocket(rec_graph, n_nodes)

        # write sampled ligands
        write_sampled_ligands(lig_pos, lig_feat, output_dir=complex_output_dir, dataset=test_dataset)

        # write keypoints to an xyz file
        kp_file = complex_output_dir / 'keypoints.xyz'
        kp_pos = kp_pos.detach().cpu()
        kp_elements = ['C' for _ in range(kp_pos.shape[0]) ]
        write_xyz_file(kp_pos, kp_elements, kp_file)


if __name__ == "__main__":

    with torch.no_grad():
        main()
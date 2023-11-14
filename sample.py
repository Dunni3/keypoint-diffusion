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
from utils import write_xyz_file, copy_graph, get_batch_idxs
from analysis.molecule_builder import make_mol_openbabel
from data_processing.make_bindingmoad_pocketfile import write_pocket_file

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, default=None, help='directory of training result for the model')
    p.add_argument('--model_file', type=str, default=None, help='Path to file containing model weights. If not specified, the most recently saved weights file in model_dir will be used')
    p.add_argument('--n_replicates', type=int, default=1)
    p.add_argument('--n_complexes', type=int, default=1)
    # p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--random', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='sampled_mols/')
    p.add_argument('--dataset', type=str, default='bindingmoad')
    p.add_argument('--idxs', type=int, nargs='+', default=[])


    p.add_argument('--visualize', action='store_true')

    args = p.parse_args()

    if args.model_file is not None and args.model_dir is not None:
        raise ValueError('only model_file or model_dir can be specified but not both')
    
    if args.dataset not in ['crossdocked', 'bindingmoad']:
        raise ValueError('unsupported dataset')

    return args

def make_reference_files(dataset_idx: int, dataset: CrossDockedDataset, output_dir: Path, remove_hydrogen: bool) -> Path:

    output_dir = output_dir / f'pocket_{dataset_idx}'
    output_dir.mkdir(exist_ok=True)

    # get original receptor and ligand files
    ref_rec_file, ref_lig_file = dataset.get_files(dataset_idx)
    ref_rec_file = Path(ref_rec_file)
    ref_lig_file = Path(ref_lig_file)

    # get filepath of new ligand and receptor files
    centered_lig_file = output_dir / ref_lig_file.name
    centered_rec_file = output_dir / ref_rec_file.name

    shutil.copy(ref_rec_file, centered_rec_file)
    shutil.copy(ref_lig_file, centered_lig_file)


    # write the pocket file 
    pocket_file = output_dir / 'pocket.pdb'
    write_pocket_file(ref_rec_file, ref_lig_file, pocket_file, cutoff=8)

    return output_dir

def write_sampled_ligands(lig_pos, lig_feat, output_dir: Path, dataset: CrossDockedDataset, name: str = None):

    lig_pos = [ arr.detach().cpu() for arr in lig_pos ]
    lig_feat = [ arr.detach().cpu() for arr in lig_feat ]

    if name is None:
        name = 'sampled_mols'

    sdf_file = output_dir / f'{name}.sdf'
    writer = Chem.SDWriter(str(sdf_file))

    for lig_idx in range(len(lig_pos)):
        element_idxs = torch.argmax(lig_feat[lig_idx], dim=1).tolist()

        atom_elements = dataset.lig_atom_idx_to_element(element_idxs)

        mol = make_mol_openbabel(lig_pos[lig_idx], atom_elements)

        # TODO: sometimes an invariant violation occurs when writing noisy molecules but I have to reproduce/figure out why this happens
        try:
            writer.write(mol)
        except:
            print(f'failed to write ligand {lig_idx} into {sdf_file}')

    writer.close()


def main():
    
    cmd_args = parse_arguments()

    # get output dir path and create the directory
    output_dir = Path(cmd_args.output_dir)
    output_dir.mkdir(exist_ok=True)

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

    # set random seeds / create rando number generator
    torch.manual_seed(cmd_args.seed)
    rng = np.random.default_rng(42)

    # create test dataset
    dataset_path = Path(args['dataset']['location']) 
    test_dataset_path = str(dataset_path / 'val.pkl')
    test_dataset = CrossDockedDataset(name='val', processed_data_file=test_dataset_path, **args['graph'], **args['dataset'])


    # get the model architecture
    try:
        architecture = args['diffusion']['architecture']
    except KeyError:
        architecture = 'egnn'

    # get rec encoder config and dynamics config
    if architecture == 'gvp':
        rec_encoder_config = args["rec_encoder_gvp"]
        dynamics_config = args['dynamics_gvp']
    elif architecture == 'egnn':
        rec_encoder_config = args["rec_encoder"]
        dynamics_config = args['dynamics']

    # get number of ligand and receptor atom features
    n_lig_feat = args['reconstruction']['n_lig_feat']
    if architecture == 'egnn':
        n_kp_feat = args["rec_encoder"]["out_n_node_feat"]
    elif architecture == 'gvp':
        n_kp_feat = args["rec_encoder_gvp"]["out_scalar_size"]

    # determine if we're using fake atoms
    try:
        use_fake_atoms = args['dataset']['max_fake_atom_frac'] > 0
    except KeyError:
        use_fake_atoms = False

    # create diffusion model
    model = LigandDiffuser(
        n_lig_feat, 
        n_kp_feat,
        processed_dataset_dir=Path(args['dataset']['location']),
        graph_config=args['graph'],
        dynamics_config=dynamics_config, 
        rec_encoder_config=rec_encoder_config, 
        rec_encoder_loss_config=args['rec_encoder_loss'],
        use_fake_atoms=use_fake_atoms,
        **args['diffusion']).to(device=device)

    # load model weights
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # get dataset indexes for complexes we are going to sample
    if cmd_args.idxs != []:
        dataset_idxs = cmd_args.idxs
    elif cmd_args.random:
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
        ref_graph, _ = test_dataset[idx]

        # move data to correct device
        ref_graph = ref_graph.to(device)

        # get batch_idxs
        batch_idxs = get_batch_idxs(ref_graph)

        if use_fake_atoms:
            ref_lig_batch_idx = torch.zeros(ref_graph.num_nodes('lig'), device=ref_graph.device)
            ref_graph = model.remove_fake_atoms(ref_graph, ref_lig_batch_idx)

        # get array specifying the number of nodes in each ligand we sample
        n_nodes = torch.ones(size=(cmd_args.n_replicates,), dtype=int, device=device)*ref_graph.num_nodes('lig')

        # get receptor keypoints
        # note the diffusion model does receptor encoding internally,
        # so for sampling this is not strictly necessary, but i would like to visualize the position of the keypoints
        ref_graph_copy = copy_graph(ref_graph, n_copies=1)[0]
        with ref_graph_copy.local_scope():
            encoded_ref_graph = model.rec_encoder(ref_graph_copy, batch_idxs)
            kp_pos = encoded_ref_graph.nodes['kp'].data['x_0']

        # sample ligands
        lig_pos, lig_feat = model.sample_given_pocket(ref_graph, n_nodes, visualize=cmd_args.visualize)

        # write sampled ligands
        if cmd_args.visualize:
            for lig_idx in range(len(lig_pos)):
                position_frames = lig_pos[lig_idx]
                feature_frames = lig_feat[lig_idx]
                name = f'lig_{lig_idx}_frames'
                write_sampled_ligands(position_frames, feature_frames, output_dir=complex_output_dir, dataset=test_dataset, name=name)
        else:
            write_sampled_ligands(lig_pos, lig_feat, output_dir=complex_output_dir, dataset=test_dataset)

        # write keypoints to an xyz file
        kp_file = complex_output_dir / 'keypoints.xyz'
        kp_pos = kp_pos.detach().cpu()
        kp_elements = ['C' for _ in range(kp_pos.shape[0]) ]
        write_xyz_file(kp_pos, kp_elements, kp_file)


if __name__ == "__main__":

    with torch.no_grad():
        main()
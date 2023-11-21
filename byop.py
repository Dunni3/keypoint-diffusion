import argparse
import math
import pickle
import shutil
import time
from pathlib import Path

import dgl
import numpy as np
import prody
import torch
import yaml
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch.nn.functional import one_hot
from tqdm import trange

from analysis.metrics import MoleculeProperties
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.pocket_minimization import pocket_minimization
from constants import aa_to_idx
from data_processing.crossdocked.dataset import ProteinLigandDataset
from data_processing.make_bindingmoad_pocketfile import PocketSelector
from data_processing.pdbbind_processing import (build_initial_complex_graph,
                                                parse_ligand,
                                                rec_atom_featurizer)
from model_setup import model_from_config
from models.ligand_diffuser import KeypointDiffusion
from utils import copy_graph, get_rec_atom_map, write_xyz_file


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('receptor_file', type=Path, help='PDB file of the receptor')
    p.add_argument('ref_ligand_file', type=Path, help='sdf file of ligand used to define the pocket')
    p.add_argument('--model_dir', type=str, default=None, help='directory of training result for the model')
    p.add_argument('--model_file', type=str, default=None, help='Path to file containing model weights. If not specified, the most recently saved weights file in model_dir will be used')
    p.add_argument('--n_ligand_atoms', type=str, default='sample', help="""number of atoms in the ligands. If "sample" (defualt), number of atoms will be samppled from training set joint distribution.
                    If "ref", number of atoms will be the same as the reference ligand. If an integer, number of atoms will be fixed to that value""")
    p.add_argument('--output_dir', type=str, default='byop_output/')
    p.add_argument('--n_mols', type=int, default=100, help='number of molecules to sample')
    p.add_argument('--max_batch_size', type=int, default=128, help='maximum feasible batch size due to memory constraints')
    p.add_argument('--seed', type=int, default=None, help='random seed as an integer. by default, no random seed is set.')

    # p.add_argument('--no_metrics', action='store_true')
    # p.add_argument('--no_minimization', action='store_true')
    p.add_argument('--ligand_only_minimization', action='store_true')
    p.add_argument('--pocket_minimization', action='store_true')
    
    args = p.parse_args()

    if args.model_file is not None and args.model_dir is not None:
        raise ValueError('only model_file or model_dir can be specified but not both')
    

    if args.model_file is None and args.model_dir is None:
        raise ValueError('one of model_file or model_dir must be specified')
    
    if args.n_ligand_atoms not in ['sample', 'ref']:
        if not args.n_ligand_atoms.isdigit():
            raise ValueError('n_ligand_atoms must be "sample", "ref", or an integer')
        args.n_ligand_atoms = int(args.n_ligand_atoms)

    return args

def make_reference_files(dataset_idx: int, dataset: ProteinLigandDataset, output_dir: Path) -> Path:

    # get original receptor and ligand files
    ref_rec_file, ref_lig_file = dataset.get_files(dataset_idx)
    ref_rec_file = Path(ref_rec_file)
    ref_lig_file = Path(ref_lig_file)

    # get filepath of new ligand and receptor files
    centered_lig_file = output_dir / ref_lig_file.name
    centered_rec_file = output_dir / ref_rec_file.name

    shutil.copy(ref_rec_file, centered_rec_file)
    shutil.copy(ref_lig_file, centered_lig_file)

    return output_dir

def write_ligands(mols, filepath: Path):
    writer = Chem.SDWriter(str(filepath))

    for mol in mols:
        writer.write(mol)

    writer.close()

def element_fixer(element: str):

    if len(element) > 1:
        element = element[0] + element[1:].lower()
    
    return element

def process_ligand_and_pocket(rec_file: Path, lig_file: Path, output_dir: Path,
                                  rec_element_map, lig_element_map,
                                  n_keypoints: int, graph_cutoffs: dict,
                                  pocket_cutoff: float, remove_hydrogen: bool = True, ca_only: bool = False):
    
    
    if rec_file.suffix == '.pdb':
        parser = PDBParser(QUIET=True)
    elif rec_file.suffix == '.mmcif':
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'unsupported receptor file type: {rec_file.suffix}, must be .pdb or .mmcif')

    rec_struct = parser.get_structure('', rec_file)

    _, lig_coords, lig_atom_features = parse_ligand(lig_file, lig_element_map, remove_hydrogen=remove_hydrogen)

    # make ligand data into torch tensors
    lig_coords = torch.tensor(lig_coords, dtype=torch.float32)

    # get residues which constitute the binding pocket
    pocket_residues = []
    for residue in rec_struct.get_residues():

        # check if residue is a standard amino acid
        is_residue = is_aa(residue.get_resname(), standard=True)
        if not is_residue:
            continue

        # get atomic coordinates of residue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])

        # check if residue is interacting with protein
        min_rl_dist = cdist(lig_coords, res_coords).min()
        if min_rl_dist < pocket_cutoff:
            pocket_residues.append(residue)

    if len(pocket_residues) == 0:
        raise ValueError(f'no valid pocket residues found.')

    if remove_hydrogen:
        atom_filter = lambda a: a.element != "H"
    else:
        atom_filter = lambda a: True

    pocket_atomres = []
    for res in pocket_residues:

        if ca_only:
            atom_list = [ res['CA'] ]
        else:
            atom_list = res.get_atoms()

        pocket_atomres.extend([(a, res) for a in atom_list if atom_filter(a) ])

    pocket_atoms, atom_residues = list(map(list, zip(*pocket_atomres)))
    res_to_idx = { res:i for i, res in enumerate(pocket_residues) }
    pocket_res_idx = list(map(lambda res: res_to_idx[res], atom_residues)) #  list containing the residue of every atom using integers to index pocket residues
    pocket_res_idx = torch.tensor(pocket_res_idx)

    pocket_coords = torch.tensor(np.array([a.get_coord() for a in pocket_atoms]))
    pocket_elements = np.array([ element_fixer(a.element) for a in pocket_atoms ])

    if ca_only:

        try:
            # get single-letter amino acid code for each residue
            res_chars = [ protein_letters_3to1[res.get_resname()] for res in pocket_residues ]

            # convert residue characters to indices
            res_idx = [ aa_to_idx[res] for res in res_chars ]
            res_idx = torch.tensor(res_idx)
        except KeyError as e:
            raise ValueError(f'unsupported residue type found: {[ res.get_resname() for res in pocket_residues ]}')

        # one-hot encode residue types
        pocket_atom_features = one_hot(res_idx, num_classes=len(aa_to_idx)).bool()

        # create an empty other_atoms_mask
        other_atoms_mask = torch.zeros(pocket_atom_features.shape[0], dtype=torch.bool)

    else:
        pocket_atom_features, other_atoms_mask = rec_atom_featurizer(rec_element_map, protein_atom_elements=pocket_elements)
        pocket_atom_features = torch.tensor(pocket_atom_features).bool()

    # remove other atoms from pocket
    pocket_coords = pocket_coords[~other_atoms_mask]
    pocket_atom_features = pocket_atom_features[~other_atoms_mask]

    # build graph
    g: dgl.DGLHeteroGraph = build_initial_complex_graph(
        rec_atom_positions=pocket_coords,
        rec_atom_features=pocket_atom_features,
        pocket_res_idx=pocket_res_idx,
        n_keypoints=n_keypoints,
        cutoffs=graph_cutoffs,
        lig_atom_positions=lig_coords,
        lig_atom_features=lig_atom_features
    )

    # save the pocket file
    pocket_file = output_dir / f'pocket.pdb'
    pocket_selector = PocketSelector(pocket_residues)
    io_object = PDBIO()
    io_object.set_structure(rec_struct)
    io_object.save(str(pocket_file), pocket_selector)

    return g


def main():
    
    args = parse_arguments()

    # get output dir path and create the directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # get filepath of config file within model_dir
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
        model_file = model_dir / 'model.pt'
    elif args.model_file is not None:
        model_file = Path(args.model_file)
        model_dir = model_file.parent
    
    # get config file
    config_file = model_dir / 'config.yml'

    # load model configuration
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}', flush=True)

    # set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # isolate dataset config
    dataset_config = config['dataset']

    # determine if we are using a Ca-only representation of the receptor
    try:
        ca_only: bool = dataset_config['ca_only']
    except KeyError:
        ca_only = False

    # construct atom typing maps
    rec_element_map, lig_element_map = get_rec_atom_map(dataset_config)
    lig_decoder = { v:k for k,v in lig_element_map.items() }


    model: KeypointDiffusion = model_from_config(config).to(device)

    # load model weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # iterate over dataset and draw samples for each pocket
    pocket_sample_start = time.time()


    # process the receptor and pocket files
    rec_file = args.receptor_file
    ref_lig_file = args.ref_ligand_file

    if not rec_file.exists() or not ref_lig_file.exists():
        raise ValueError('receptor or reference ligand file does not exist')
    
    ref_graph: dgl.DGLHeteroGraph = process_ligand_and_pocket(
                                rec_file, ref_lig_file, output_dir,
                                rec_element_map=rec_element_map,
                                lig_element_map=lig_element_map,
                                n_keypoints=config['graph']['n_keypoints'],
                                graph_cutoffs=config['graph']['graph_cutoffs'],
                                pocket_cutoff=dataset_config['pocket_cutoff'], 
                                remove_hydrogen=dataset_config['remove_hydrogen'],
                                ca_only=ca_only)

    # TODO: how should/could we handle fake atoms? do we need to worry about it?
    # none of the trained models actually use fake atoms, so this is not a problem for now
    try:
        use_fake_atoms = config['dataset']['max_fake_atom_frac'] > 0
    except KeyError:
        use_fake_atoms = False
    if use_fake_atoms:
        raise NotImplementedError('fake atoms are not supported')
    #     ref_lig_batch_idx = torch.zeros(ref_graph.num_nodes('lig'), device=ref_graph.device)
    #     ref_graph = model.remove_fake_atoms(ref_graph, ref_lig_batch_idx)

    ref_graph = ref_graph.to(device)

    # get the number of nodes in the binding pocket
    n_rec_nodes = ref_graph.num_nodes('rec')
    n_rec_nodes = torch.tensor([n_rec_nodes], device=device)

    # encode the receptor
    ref_graph = model.encode_receptors(ref_graph)

    # compute initial ligand COM
    # TODO: add an option for user-provided initial ligand COM
    ref_init_lig_com = dgl.readout_nodes(ref_graph, ntype='lig', feat='x_0', op='mean')

    n_samplings = math.ceil(args.n_mols / args.max_batch_size)
    n_samplings += 1
    
    pocket_raw_mols = []
    for attempt_idx in range(n_samplings):

        n_mols_needed = args.n_mols - len(pocket_raw_mols)
        n_mols_to_generate = math.ceil( n_mols_needed / 0.99 ) # account for the fact that only ~99% of generated molecules are valid
        batch_size = min(n_mols_to_generate, args.max_batch_size)

        # compute the number of ligand atoms in each generated molecule
        if args.n_ligand_atoms == 'sample':
            atoms_per_lig = model.lig_size_dist.sample(n_rec_nodes, batch_size).to(device).flatten()
        elif args.n_ligand_atoms == 'ref':
            atoms_per_lig = torch.tensor([ref_graph.num_nodes('lig')]*batch_size, device=device)
        else:
            atoms_per_lig = torch.tensor([args.n_ligand_atoms]*batch_size, device=device)

        # copy the reference graph out batch_size times
        g_batch = copy_graph(ref_graph, batch_size, lig_atoms_per_copy=atoms_per_lig)
        g_batch = dgl.batch(g_batch)

        # copy the ref_lig_com out batch_size times
        init_lig_com_batch = ref_init_lig_com.repeat(batch_size, 1)

        # sample ligand atom positions/features
        with g_batch.local_scope():
            batch_lig_pos, batch_lig_feat = model.sample_from_encoded_receptors(
                g_batch,  
                init_lig_pos=init_lig_com_batch)

        # convert positions/features to rdkit molecules
        for lig_idx, (lig_pos_i, lig_feat_i) in enumerate(zip(batch_lig_pos, batch_lig_feat)):

            # convert lig atom features to atom elements
            element_idxs = torch.argmax(lig_feat_i, dim=1).tolist()
            atom_elements = [ lig_decoder[idx] for idx in element_idxs ]

            # build molecule
            mol = build_molecule(lig_pos_i, atom_elements, add_hydrogens=False, sanitize=True, largest_frag=True, relax_iter=0)

            if mol is not None:
                pocket_raw_mols.append(mol)

        # stop generating molecules if we've made enough
        if len(pocket_raw_mols) >= args.n_mols:
            break

    pocket_sample_time = time.time() - pocket_sample_start

    # save pocket sample time
    with open(output_dir / 'sample_time.txt', 'w') as f:
        f.write(f'{pocket_sample_time:.2f}')

    # print the sampling time
    print(f'sampling time: {pocket_sample_time:.2f}')

    # print the sampling time per molecule
    print(f'sampling time per molecule: {pocket_sample_time/len(pocket_raw_mols):.2f}')

    # write the reference files to the pocket dir
    ref_files_dir = output_dir / 'reference_files'
    ref_files_dir.mkdir(exist_ok=True)
    shutil.copy(ref_lig_file, ref_files_dir)
    shutil.copy(rec_file, ref_files_dir)

    # give molecules a name
    for idx, mol in enumerate(pocket_raw_mols):
        mol.SetProp('_Name', f'lig_idx_{idx}')

    # write the ligands to the pocket dir
    write_ligands(pocket_raw_mols, output_dir / 'raw_ligands.sdf')

    # ligand-only minimization
    if args.ligand_only_minimization:
        pocket_lomin_mols = []
        for raw_mol in pocket_raw_mols:
            minimized_mol = process_molecule(Chem.Mol(raw_mol), add_hydrogens=True, relax_iter=200)
            if minimized_mol is not None:
                pocket_lomin_mols.append(minimized_mol)
        ligands_file = output_dir / 'minimized_ligands.sdf'
        write_ligands(pocket_lomin_mols, ligands_file)

    # pocket-only minimization
    if args.pocket_minimization:
        input_mols = [ Chem.Mol(raw_mol) for raw_mol in pocket_raw_mols ]
        pocket_file = output_dir / f'pocket.pdb'
        pocket_pmin_mols, rmsd_df = pocket_minimization(pocket_file, input_mols, add_hs=True)
        ligands_file = output_dir / 'pocket_minimized_ligands.sdf'
        write_ligands(pocket_pmin_mols, ligands_file)
        rmsds_file = output_dir / 'pocket_min_rmsds.csv'
        rmsd_df.to_csv(rmsds_file, index=False)


    # get keypoint positions
    keypoint_positions = ref_graph.nodes['kp'].data['x_0']
    
    # write keypoints to an xyz file
    kp_file = output_dir / 'keypoints.xyz'
    kp_elements = ['C' for _ in range(keypoint_positions.shape[0]) ]
    write_xyz_file(keypoint_positions, kp_elements, kp_file)

if __name__ == "__main__":

    with torch.no_grad():
        main()
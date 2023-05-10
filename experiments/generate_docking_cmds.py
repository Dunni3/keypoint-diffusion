import argparse
from pathlib import Path
from rdkit import Chem
import numpy as np


# change this script to operate on ff-minimized ligands
# change this script to generate a list of commands from a list of pre-generated pocket pairs

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--test_result_dir', type=str, help='directory of results from testing code')
    p.add_argument('--n_pairs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_atom_diff', type=int, default=1)

    p.add_argument('--to_pairs_file', type=str, default=None)
    p.add_argument('--from_pairs_file', type=str, default=None)
    p.add_argument('--preminimized_ligs', action='store_true')
    
    args = p.parse_args()

    return args

def parse_pocket_dir(pocket_dir: Path, preminimized: bool):

        pocket_dir = pocket_dir.resolve()
        pocket_files = list(pocket_dir.iterdir())
        # pocket_files = [ x.resolve() for x in pocket_files ]

        # find the receptor pdb file within this directory
        rec_pdb_file = None
        for pocket_file in pocket_files:
            if pocket_file.suffix == '.pdb':
                rec_pdb_file = pocket_file
                break
        
        if rec_pdb_file is None:
            raise ValueError(f'no pdb file found in {pocket_dir}')
        
        # find the file for the reference ligand
        ref_ligand_file_name = '_'.join(rec_pdb_file.name.split('_')[:-1]) + '.sdf'
        ref_ligand_file = pocket_dir / ref_ligand_file_name

        if not ref_ligand_file.exists():
            raise ValueError(f'reference ligand file not found: {ref_ligand_file}')
        
        if preminimized:
            generated_ligands_file = pocket_dir / f'{pocket_dir.name}_ligands_rec_uff.sdf.gz'
        else:
            generated_ligands_file = pocket_dir / f'{pocket_dir.name}_ligands.sdf'

        return rec_pdb_file, ref_ligand_file, generated_ligands_file

def get_dst_lig_idxs(src_lig_idxs: np.ndarray, atoms_per_ligand: np.ndarray, ref_fps: np.ndarray, base_atom_diff: int):
    selected_dst_idxs = []
    for src_lig_idx in src_lig_idxs:

        n_atoms_src = atoms_per_ligand[src_lig_idx] # the number of atoms in the source ligand
        n_atom_diff = base_atom_diff

        while True:
            # find ligands who have n_atoms within +/- n_atom_diff of n_atoms_src
            dst_lig_candidate_idxs = np.where(
                (atoms_per_ligand >= (n_atoms_src - n_atom_diff)) & 
                (atoms_per_ligand <= (n_atoms_src + n_atom_diff)))[0]

            # remove ligands which have already been selected as either source or destination ligands
            has_been_selected_mask = np.isin(dst_lig_candidate_idxs, src_lig_idxs) | np.isin(dst_lig_candidate_idxs, selected_dst_idxs)
            dst_lig_candidate_idxs = dst_lig_candidate_idxs[~has_been_selected_mask]

            # confirm that there are a non-zero number of candidate destination ligands remaining
            inc_diff = dst_lig_candidate_idxs.shape[0] == 0
            if inc_diff and has_been_selected_mask.shape[0] == atoms_per_ligand.shape[0]:
                raise ValueError('impossible to find destination ligand, likely cause is that n_pairs argument is too large.')
            elif inc_diff:
                n_atom_diff += 1
            else:
                break

        # compute tanimoto coefficients between the source ligand and the candidate destination ligands
        candidate_dst_fps = [ref_fps[idx] for idx in dst_lig_candidate_idxs ]
        similarities = Chem.DataStructs.BulkTanimotoSimilarity(ref_fps[src_lig_idx], candidate_dst_fps)
        dst_lig_idx = dst_lig_candidate_idxs[np.argmin(similarities)] # select most dissimilar ligand
        selected_dst_idxs.append(dst_lig_idx)

        dst_lig_atoms = atoms_per_ligand[dst_lig_idx]
        print(f'src lig atoms: {n_atoms_src} --> dist lig atoms: {dst_lig_atoms}')

    return np.array(selected_dst_idxs)

def compute_pocket_pairs(sampled_mols_dir: Path, preminimized: bool):

    pocket_files = [ parse_pocket_dir(pocket_dir, preminimized=preminimized) for pocket_dir in sampled_mols_dir.iterdir() ]

    rec_pdb_files, ref_ligand_files, generated_ligands_files = list(zip(*pocket_files))
    
    # convert reference ligand files to rdkit molecules
    ref_ligands = [ Chem.MolFromMolFile(str(x), sanitize=False) for x in ref_ligand_files ]

    # find which reference ligands could not be converted into rdkit moleclues and filter these ligands from the sets of receptors/ref ligand/generated ligands files
    none_map = [ mol is None for mol in ref_ligands ]
    rec_pdb_files = [ x for i,x in enumerate(rec_pdb_files) if not none_map[i] ]
    ref_ligand_files = [ x for i,x in enumerate(ref_ligand_files) if not none_map[i] ]
    generated_ligands_files = [ x for i,x in enumerate(generated_ligands_files) if not none_map[i] ]
    ref_ligands = [ x for x in ref_ligands if x is not None ]

    # get number of atoms in each ligand
    n_atoms = [ mol.GetNumAtoms() for mol in ref_ligands ]
    n_atoms = np.array(n_atoms)

    # choose which ligands from the test set will be the "source ligand" for each pair
    src_lig_idxs = rng.choice(np.arange(n_atoms.shape[0]), size=args.n_pairs, replace=False)

    # get fingerprints for all ref ligands
    ref_fps = [ Chem.RDKFingerprint(mol) for mol in ref_ligands ]

    # choose which ligands from the test set will be the "destination" ligand for each pair
    dst_lig_idxs = get_dst_lig_idxs(src_lig_idxs, n_atoms, ref_fps, args.n_atom_diff)

    return rec_pdb_files, ref_ligand_files, generated_ligands_files, src_lig_idxs, dst_lig_idxs

def pairs_from_pair_file(pair_file: str, sampled_mols_dir: Path, preminimized: bool):
    
    # read pairs file
    pocket_pair_names = []
    with open(pair_file, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            if len(split_line) == 0:
                continue
            pocket_pair_names.append(split_line)
    
    src_pocket_names, dst_pocket_names = list(map(list, zip(*pocket_pair_names)))

    pocket_names = src_pocket_names + dst_pocket_names
    src_lig_idxs = list(range(len(src_pocket_names)))
    dst_lig_idxs = list(range(len(src_pocket_names), len(pocket_names)))

    pocket_files = [ parse_pocket_dir(sampled_mols_dir / pocket_name, preminimized=preminimized) for pocket_name in pocket_names ]
    rec_pdb_files, ref_ligand_files, generated_ligands_files = list(zip(*pocket_files))
    return rec_pdb_files, ref_ligand_files, generated_ligands_files, src_lig_idxs, dst_lig_idxs

if __name__ == "__main__":

    args = parse_arguments()

    rng = np.random.default_rng(args.seed)

    test_result_dir = Path(args.test_result_dir).resolve()
    sampled_mols_dir = test_result_dir / 'sampled_mols'

    if args.from_pairs_file is None:
        rec_pdb_files, ref_ligand_files, generated_ligands_files, src_lig_idxs, dst_lig_idxs = compute_pocket_pairs(sampled_mols_dir, preminimized=args.preminimized_ligs)
    else:
        rec_pdb_files, ref_ligand_files, generated_ligands_files, src_lig_idxs, dst_lig_idxs = pairs_from_pair_file(args.from_pairs_file, sampled_mols_dir, preminimized=args.preminimized_ligs)

    # determine/create directories for output files associated with this experiment
    output_dir = test_result_dir / 'crossdock_experiment'
    gnina_output_dir = output_dir / 'gnina_output'
    output_dir.mkdir(exist_ok=True)
    gnina_output_dir.mkdir(exist_ok=True)

    # a list to collect the gnina docking commands
    docking_cmds = []


    # for each pair of (src_pocket, dst_pocket) we will create a gnina command to do each of the following
    # docking ligands generated for src_pocket into src_pocket
    # dock ligands generated for src_pocket into dst_pocket
    # dock ligands generated for dst_pocket into dst_pocket
    # dock ligands generated for dst_pocket into src_pocket
    for src_idx, dst_idx in zip(src_lig_idxs, dst_lig_idxs):

        for lig_idx in [src_idx, dst_idx]:
            for rec_idx in [src_idx, dst_idx]:

                # get the pocket which the generated ligands and receptors are coming from, respectively
                lig_pocket = rec_pdb_files[lig_idx].parent.name
                rec_pocket = rec_pdb_files[rec_idx].parent.name

                # determine the name of the gnina output file
                docking_result_file = gnina_output_dir / f'rec_{rec_pocket}_lig_{lig_pocket}.sdf.gz'

                # create/record the gnina command
                cmd = f"gnina -r {rec_pdb_files[rec_idx]} -l {generated_ligands_files[lig_idx]} --autobox_ligand={ref_ligand_files[rec_idx]} -o {docking_result_file} --cpu 8"
                docking_cmds.append(cmd)
    
    # write gnina commands to a file
    docking_cmds_file = output_dir / 'docking_cmds.txt'
    with open(docking_cmds_file, 'w') as f:
        f.write('\n'.join(docking_cmds))

    # write pairs file
    if args.to_pairs_file is not None:

        with open(args.to_pairs_file, 'w') as f:
            for src_idx, dst_idx in zip(src_lig_idxs, dst_lig_idxs):

                src_pocket_name = rec_pdb_files[src_idx].parent.name
                dst_pocket_name = rec_pdb_files[dst_idx].parent.name

                f.write(f'{src_pocket_name},{dst_pocket_name}\n')
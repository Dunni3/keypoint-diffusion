import argparse
from pathlib import Path
from rdkit import Chem
import numpy as np

DEFAULT_MINIMIZATION_SCRIPT = str(Path('minimize_ligands.py').resolve())

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--test_result_dir', type=str, help='directory of results from testing code')
    p.add_argument('--cmd_file', type=str, help='path to file where commands will be written', default=None)

    p.add_argument('--minimization_script', type=str, default=DEFAULT_MINIMIZATION_SCRIPT)
    
    args = p.parse_args()

    return args

def parse_pocket_dir(pocket_dir: Path):

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
        
        generated_ligands_file = pocket_dir / f'{pocket_dir.name}_ligands.sdf'

        return rec_pdb_file, ref_ligand_file, generated_ligands_file

if __name__ == "__main__":

    args = parse_arguments()

    # get directory which has sampled molecules
    test_result_dir = Path(args.test_result_dir).resolve()
    sampled_mols_dir = test_result_dir / 'sampled_mols'

    # get receptor pdb files, reference ligand files, and generated ligand files for every pocket in the test set
    pocket_files = [ parse_pocket_dir(pocket_dir) for pocket_dir in sampled_mols_dir.iterdir() ]
    rec_pdb_files, ref_ligand_files, generated_ligands_files = list(zip(*pocket_files))

    # for every pocket, genreate a command for minimizing the generated ligands within the receptor
    cmds = []
    for rec_file, lig_file in zip(rec_pdb_files, generated_ligands_files):
        cmd = f'python {args.minimization_script} --rec_file={rec_file} --lig_file={lig_file}'
        cmds.append(cmd)

    # write commands to output file
    if args.cmd_file is None:
        args.cmd_file = Path(args.test_result_dir).resolve() / 'minimize_cmds.txt'

    with open(args.cmd_file, 'w') as f:
         f.write('\n'.join(cmds))

    
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser('Generate docking commands for gnina')
    p.add_argument('sampled_mols_dir', type=Path)
    p.add_argument('--cpu', type=int, default=1, help='Number of cpus to use for docking')
    
    # add an argument for the output file
    p.add_argument('--output_file', type=Path, default=Path('docking_cmds.txt'), help='File to write the commands to')

    # add a flag for whether to do full docking or just minimization w.r.t. vina scoring function
    p.add_argument('--minimize', action='store_true', help='whether to do full docking or just minimization w.r.t. vina scoring function')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    output_cmds = []
    for pocket_dir in args.sampled_mols_dir.iterdir():


        # get the filepath of generated + ff-minimized ligands
        gen_lig_file = pocket_dir / 'pocket_minimized_ligands.sdf'
        
        # get ref lig file
        reference_file_dir = pocket_dir / 'reference_files'
        try:
            ref_lig_file = list(reference_file_dir.glob('[!.]*.sdf'))[0]
        except IndexError:
            print(f'No reference ligand found for {pocket_dir}, using the first raw generated ligand instead')
            ref_lig_file = pocket_dir / 'raw_ligands.sdf'


        # get pocket file
        pocket_file = pocket_dir / 'pocket.pdb'

        if args.minimize:
            ref_dock_output = pocket_dir / 'ref_ligand_gnina_minimized.sdf'
            gen_dock_output = pocket_dir / 'gen_ligands_gnina_minimized.sdf'
        else:
            ref_dock_output = pocket_dir / 'ref_ligand_docked.sdf'
            gen_dock_output = pocket_dir / 'gen_ligands_docked.sdf'

        cmd_ref = f'gnina -r {pocket_file} -l {ref_lig_file} --autobox_ligand {ref_lig_file} -o {ref_dock_output} --cpu {args.cpu} {minimize_cmd}'
        cmd_gen = f'gnina -r {pocket_file} -l {gen_lig_file} --autobox_ligand {ref_lig_file} -o {gen_dock_output} --cpu {args.cpu} {minimize_cmd}'
        output_cmds.append(f'{cmd_ref};{cmd_gen}\n')

    with open(args.output_file, 'w') as f:
        f.write(''.join(output_cmds))
from pathlib import Path
import argparse
from collections import defaultdict

def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument('sampled_mols_dir', type=Path)
    p.add_argument('--minimization_script', type=Path, default=Path('analysis/pocket_minimization.py'))
    p.add_argument('--cpus', type=int, default=1)
    p.add_argument('--redo', action='store_true')
    p.add_argument('--cmd_file', type=Path, default=Path('min_cmds.txt'))

    args = p.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    stat_counter = defaultdict(int)

    cmds = []
    for pocket_dir in args.sampled_mols_dir.iterdir():

        stat_counter['n_pocket_dirs'] += 1

        # find relevant files
        minimized_ligands_file = pocket_dir / 'pocket_minimized_ligands.sdf'
        rec_file = pocket_dir / 'pocket.pdb'
        lig_file = pocket_dir / 'raw_ligands.sdf'
        running_file = pocket_dir / 'min_running'

        # skip pockets for which minimization has already been done
        if minimized_ligands_file.exists():
            stat_counter['pockets_already_minimized'] += 1
            if not args.redo:
                continue

        # skip pockets that don't have ligands yet
        if not lig_file.exists():
            stat_counter['pockets_without_ligands'] += 1
            continue


        # skip pockets that already have a minimization job running on them
        if running_file.exists():
            stat_counter['pockets_already_running'] += 1
            continue

        # construct command
        cmd = f"python {args.minimization_script} --rec_file {rec_file} --lig_file {lig_file} --cpus {args.cpus}\n"
        cmds.append(cmd)

    with open(args.cmd_file, 'w') as f:
        f.write(''.join(cmds))

    for key in ['pockets_already_minimized', 'pockets_without_ligands', 'pockets_already_running']:
        print(f"{key} = {stat_counter[key]}/{stat_counter['n_pocket_dirs']}")

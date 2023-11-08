import argparse
import pickle
from pathlib import Path

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('base_cmd_file', type=Path)
    p.add_argument('--dataset_idxs_file', type=Path, default=Path('val_subset/val_idxs.pkl'))
    p.add_argument('--output_cmd_file', type=Path, default=Path('test_cmds_parallel.txt'))
    p.add_argument('--lines', type=int, nargs='+', default=[])
    p.add_argument('--filenames_file', type=Path, default=None)
    args = p.parse_args()
    return args

if __name__ == "__main__":

    args = parse_arguments()

    with open(args.base_cmd_file, 'r') as f:
        base_cmd_lines = [ line.strip() for line in f ]

    if args.lines == []:
        args.lines = list(range(len(base_cmd_lines)))

    base_cmd_lines = [ x for i,x in enumerate(base_cmd_lines) if i in args.lines ]

    if args.filenames_file is not None:
        with open(args.filenames_file, 'rb') as f:
            filenames_dict = pickle.load(f)
        dataset_idxs = list(range(len(filenames_dict['lig_files'])))
    else:
        with open(args.dataset_idxs_file, 'rb') as f:
            dataset_idxs = pickle.load(f)
    
    cmds = []
    for base_cmd in base_cmd_lines:

        for dataset_idx in dataset_idxs:
            cmd = f'{base_cmd} --dataset_idx {dataset_idx}\n'
            cmds.append(cmd)

    with open(args.output_cmd_file, 'w') as f:
        f.write(''.join(cmds))
        
    

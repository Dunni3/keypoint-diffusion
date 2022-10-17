from pathlib import Path
import pdb
import sys
import re
from tqdm import tqdm

data_dir = Path(sys.argv[1]) # path to root of PDBbind datset

water_patten = re.compile(r"HETATM +\d+ .+ +HOH")

# count number of files in the directory
n_dirs = 0
for pdb_id_dir in data_dir.iterdir():
    n_dirs += 1


pbar = tqdm(total=n_dirs)
for pdb_id_dir in data_dir.iterdir():
    # skip things that are not data directories
    if pdb_id_dir.name in ["index", "readme"] or not pdb_id_dir.is_dir():
        continue

    protein_file = pdb_id_dir / f'{pdb_id_dir.name}_protein.pdb'

    # find all lines of the file that are not waters
    lines = []
    with open(protein_file, 'r') as f:
        for line in f:
            if re.match(water_patten, line) is None: # if this line is not declaring a water molecule
                lines.append(line)
    
    # write out the pdb file without water
    output_file = pdb_id_dir / f'{pdb_id_dir.name}_protein_nowater.pdb'
    with open(output_file, 'w') as f:
        f.write(''.join(lines))

    # update progress bar
    pbar.update(1)
    

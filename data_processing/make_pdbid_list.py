from pathlib import Path

pdbbind_index_path = Path('/home/ian/projects/mol_diffusion/ligdiff/data/PDBbind/index/INDEX_refined_name.2020')
output_index_path = '/home/ian/projects/mol_diffusion/ligdiff/data/PDBbind_processed/train_index.txt'

pdb_ids = []

pdbbind_index = open(pdbbind_index_path, 'r')
output_index = open(output_index_path, 'w')


# iterate over lines in the original PDBbind index
for line in pdbbind_index:
    if line.startswith('#'):
        continue
    pdb_id = line[:4]

    # write just the PDB ID to the output index file
    output_index.write(pdb_id + '\n')
    


pdbbind_index.close()
output_index.close()

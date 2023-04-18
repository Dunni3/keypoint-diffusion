import argparse
from pathlib import Path
import shutil
import rdkit.Chem as Chem

from analysis.molecule_builder import process_molecule

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--diffsbdd_test_results', type=str, required=True, help='directory containing the output of DiffSBDD/test.py')
    # p.add_argument('--test_set_dir', type=str, help='filepath of directory containing DiffSBDD processed crossdocked test set', required=True)
    p.add_argument('--output_dir', type=str, required=True, help='directory to write the rearranged test results into')
    p.add_argument('--example_output_dir', required=True, help='filepath of directory containing results from ligdiff/test_crossdocked.py')

    args = p.parse_args()
    return args

def extract_largest_frag(sdf_file_path: Path):
    mols_processed = []
    mols_unprocessed = [ m for m in Chem.SDMolSupplier(str(sdf_file_path), sanitize=False) ]
    for mol in mols_unprocessed:
        mol = process_molecule(mol, add_hydrogens=True, sanitize=True, largest_frag=True)
        if mol is not None:
            mols_processed.append(mol)

    with Chem.SDWriter(str(sdf_file_path)) as w:
        for mol in mols_processed:
            w.write(mol)

if __name__ == "__main__":
    args = parse_args()

    # construct output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir()
    output_sampled_mols_dir = output_dir / 'sampled_mols'
    output_sampled_mols_dir.mkdir()

    # get filepath of DiffSBDD generated molecuels
    diffsbdd_result_dir = Path(args.diffsbdd_test_results)
    diffsbdd_raw_results = diffsbdd_result_dir / 'raw'

    example_dir = Path(args.example_output_dir) / 'sampled_mols'
    for example_pocket_dir in example_dir.iterdir():
        
        # find receptor file for this pocket
        pdb_files = [ file for file in example_pocket_dir.iterdir() if file.suffix == '.pdb' ]
        assert len(pdb_files) == 1
        rec_file = pdb_files[0]

        # find reference ligand file for this pocket
        pdb_id = rec_file.name.split('_')[0]
        ref_lig_files = [ file for file in example_pocket_dir.iterdir() if file.suffix == '.sdf' and example_pocket_dir.name not in file.name ]
        assert len(ref_lig_files) == 1
        ref_lig_file = ref_lig_files[0]

        # find the corresponding DiffSBDD generated ligands
        diffsbdd_pocket_name = rec_file.stem.replace('_', '-')
        matched_lig_files = [ file for file in diffsbdd_raw_results.iterdir() if diffsbdd_pocket_name in file.name ]
        if len(matched_lig_files) == 0:
            continue
        assert len(matched_lig_files) == 1 # this will break if the test set contains duplicate receptors
        diffsbdd_lig_file = matched_lig_files[0]

        # construct the directory that will contain DiffSBDD generated ligands in a format that works with our evaluation/analysis scripts
        output_pocket_dir = output_sampled_mols_dir / example_pocket_dir.name
        output_pocket_dir.mkdir()

        # construct filepath for this pocket's ligands
        new_lig_file = output_pocket_dir / f'{example_pocket_dir.name}_ligands.sdf'

        shutil.copy(diffsbdd_lig_file, new_lig_file) # copy DiffSBDD generated ligands into output dir
        shutil.copy(rec_file, output_pocket_dir) # copy receptor file into output dir
        shutil.copy(ref_lig_file, output_pocket_dir) # copy referenace ligand file into output dir

        # retain only largest fragment from ligands
        extract_largest_frag(new_lig_file)
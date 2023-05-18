import argparse
from pathlib import Path
import rdkit.Chem as Chem
import pickle

from analysis.metrics import MoleculeProperties

def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument('sampled_mols_dir', type=Path)

    args = p.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    ligands = []
    pocket_dirs = []
    for pocket_dir in args.sampled_mols_dir.iterdir():

        pocket_ligands_file = pocket_dir / 'raw_ligands.sdf'
        pocket_ligands = list( Chem.SDMolSupplier(str(pocket_ligands_file), sanitize=False) )
        pocket_ligands = [ x for x in pocket_ligands if x is not None]
        pocket_dirs.append(pocket_dir)
        ligands.append(pocket_ligands)

    mol_metrics = MoleculeProperties()
    qed, sa, logp, lipinski, per_pocket_diversity = \
        mol_metrics.evaluate(ligands)
    
    # write metrics into a big wile
    metrics_dict = {
        'qed': qed, 'sa':sa, 'logp':logp, 'diversity': per_pocket_diversity, 'pocket_dirs': pocket_dirs }

    with open(args.sampled_mols_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics_dict, f)
    

from rdkit.Chem import AllChem as Chem
# import numpy as np
from pathlib import Path
import gzip
import argparse
import pandas as pd
from multiprocessing import Pool
import atexit

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--rec_file', type=str, help='filepath of receptor')
    p.add_argument('--lig_file', type=str, help='SDF file containing ligand(s)')
    p.add_argument('--cpus', type=int, default=1)
    p.add_argument('--output_file', type=str, help='output file path for ligands. defaults to pocket_minimized_ligands.sdf', default=None)
    
    args = p.parse_args()

    return args

def compute_rmsd(mol1, mol2):
    return Chem.CalcRMS(mol1, mol2)

def pocket_minimization(pocket_file: Path, ligands = None, add_hs=False, cpus=1):
    # load receptor and add hydrogens
    rec = Chem.MolFromPDBFile(str(pocket_file))
    rec = Chem.AddHs(rec, addCoords=True)

    # add hydrogens to ligands if necessary
    if add_hs:
        ligands = [ Chem.AddHs(lig, addCoords=True) for lig in ligands ]

    # minimize and compute rmsds for all ligands
    rmsd_table_rows = []
    minimized_ligands = []
    if cpus == 1:
        for lig_idx, ref_lig in enumerate(ligands):


            result = minimize_ligand(ref_lig, lig_idx, rec)

            if result is None:
                continue

            before_energy, after_energy, row, lig = result

            rmsd_table_rows.append(row)
            minimized_ligands.append(lig)
    else:

        args = [ (ref_lig, lig_idx, rec) for lig_idx, ref_lig in enumerate(ligands) ]
        with Pool(cpus) as p:
            results = p.starmap(minimize_ligand, args)

        for result in results:

            if result is None:
                continue

            before_energy, after_energy, row, lig = result
            rmsd_table_rows.append(row)
            minimized_ligands.append(lig)
        
    rmsd_df = pd.DataFrame(rmsd_table_rows)
    return minimized_ligands, rmsd_df

def minimize_ligand(ref_lig, lig_idx, rec):

    # create a copy of the original ligand
    lig = Chem.Mol(ref_lig)

    if lig is None:
        return None

    complex = Chem.CombineMols(rec,lig)
    Chem.SanitizeMol(complex)

    try:
        ff = Chem.UFFGetMoleculeForceField(complex,ignoreInterfragInteractions=False)
    except Exception as e:
        return None
    
    before_energy = ff.CalcEnergy()

    for p in range(rec.GetNumAtoms()):
        ff.AddFixedPoint(p)

    # documentation for this function call, incase we want to play with number of minimization steps or record whether it was successful: https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html#rdkit.ForceField.rdForceField.ForceField.Minimize
    try:
        ff.Minimize(maxIts=400)
    except Exception as e:
        return None

    after_energy = ff.CalcEnergy()

    cpos = complex.GetConformer().GetPositions()
    conf = lig.GetConformer()
    for (i,xyz) in enumerate(cpos[-lig.GetNumAtoms():]):
        conf.SetAtomPosition(i,xyz)

    # compute rmsd between original and minimized ligand
    rmsd = compute_rmsd(ref_lig, lig)

    row = {'lig_idx': lig_idx, 'rmsd': rmsd}

    # save the minimized ligand
    lig.SetProp('_Name', f'lig_idx_{lig_idx}')
    return before_energy, after_energy, row, lig

def remove_running_file(running_file: Path):
    running_file.unlink()

if __name__ == "__main__":

    args = parse_arguments()

    running_file = Path(args.lig_file).parent / 'min_running'
    running_file.touch()
    atexit.register(remove_running_file, running_file)

    ligands = list( Chem.SDMolSupplier(args.lig_file, sanitize=False) )
    minimized_ligands, rmsd_df = pocket_minimization(args.rec_file, ligands, add_hs=True, cpus=args.cpus)

    # write minimized ligands out
    if args.output_file is None:
        lig_file_path = Path(args.lig_file)
        outfile_path = lig_file_path.parent / f"pocket_minimized_ligands.sdf"
    else:
        outfile_path = args.output_file

    outfile = open(outfile_path, 'w')

    out = Chem.SDWriter(outfile)
    for lig in minimized_ligands:
        out.write(lig)
    out.close()
    outfile.close()

    # write rmsd table out as a csv
    rmsd_file_path = Path(args.lig_file).parent / 'pocket_min_rmsds.csv'
    rmsd_df.to_csv(rmsd_file_path, index=False)
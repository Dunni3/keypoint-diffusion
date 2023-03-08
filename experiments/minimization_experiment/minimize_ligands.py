from rdkit.Chem import AllChem as Chem
# import numpy as np
from pathlib import Path
import gzip
import argparse

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--rec_file', type=str, help='filepath of receptor')
    p.add_argument('--lig_file', type=str, help='SDF file containing ligand(s)')
    p.add_argument('--output_file', type=str, help='output file path. defaults to {--lig_file}_rec_uff.sdf.gz', default=None)
    
    args = p.parse_args()

    return args

if __name__ == "__main__":

    args = parse_arguments()

    rec = Chem.MolFromPDBFile(args.rec_file)
    rec = Chem.AddHs(rec, addCoords=True)

    # if lname.endswith('.gz'):
    #     lig = next(Chem.ForwardSDMolSupplier(gzip.open(lname),sanitize=False))
    # else:    
    #     lig = next(Chem.SDMolSupplier(sys.argv[2],sanitize=False))

    ligands = list( Chem.SDMolSupplier(args.lig_file, sanitize=False) )

    minimized_ligands = []
    for lig_idx, lig in enumerate(ligands):

        print(f'minimizing {lig_idx+1}/{len(ligands)}', flush=True)

        if lig is None:
            continue

        complex = Chem.CombineMols(rec,lig)
        Chem.SanitizeMol(complex)

        try:
            ff = Chem.UFFGetMoleculeForceField(complex,ignoreInterfragInteractions=False)
        except Exception as e:
            print(e)
            print(f'Failed to construct ligand forcefield {lig_idx}', flush=True)
            continue

        print("Before:",ff.CalcEnergy(), flush=True)

        for p in range(rec.GetNumAtoms()):
            ff.AddFixedPoint(p)

        # documentation for this function call, incase we want to play with number of minimization steps or record whether it was successful: https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html#rdkit.ForceField.rdForceField.ForceField.Minimize
        try:
            ff.Minimize(maxIts=400)
        except Exception as e:
            print(e)
            print(f'Failed to minimize ligand {lig_idx}', flush=True)
            continue

        print("After:",ff.CalcEnergy())

        cpos = complex.GetConformer().GetPositions()
        conf = lig.GetConformer()
        for (i,xyz) in enumerate(cpos[-lig.GetNumAtoms():]):
            conf.SetAtomPosition(i,xyz)

        lig.SetProp('_Name', f'lig_idx_{lig_idx}')
        minimized_ligands.append(lig)

    # write minimized ligands out
    if args.output_file is None:
        lig_file_path = Path(args.lig_file)
        outfile_path = lig_file_path.parent / f"{lig_file_path.name.split('.')[0]}_rec_uff.sdf.gz"
    else:
        outfile_path = args.output_file

    outfile = gzip.open(outfile_path, 'wt')

    out = Chem.SDWriter(outfile)
    for lig in minimized_ligands:
        out.write(lig)
    out.close()
    outfile.close()
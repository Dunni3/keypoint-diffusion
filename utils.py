import openbabel
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdDetermineBonds
import tempfile

# this is taken from DiffSBDD, minor modification to return the file contents without writing to disk if filename=None 
def write_xyz_file(coords, atom_types, filename = None):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"

    if filename is None:
        return out
    
    with open(filename, 'w') as f:
        f.write(out)

# this is taken from DiffSBDD's make_mol_openbabel, i've adapted their method to do bond determination using rdkit
# useful blog post: https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html
# update, i wasn't able to make this work, so
# TODO: make this work
def make_mol_rdkit(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    # Write xyz file
    xyz_file_contents = write_xyz_file(positions, atom_types)

    # get rdkit mol object from xyz file
    raw_mol = Chem.MolFromXYZBlock(xyz_file_contents)

    # find bonds, without determining bond-orders
    conn_mol = Chem.Mol(raw_mol)
    conn_mol = rdDetermineBonds.DetermineConnectivity(conn_mol)

    # Convert to sdf file with openbabel
    # openbabel will add bonds
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")
    ob_mol = openbabel.OBMol()
    obConversion.ReadFile(ob_mol, tmp_file)

    obConversion.WriteFile(ob_mol, tmp_file)

    # Read sdf file with RDKit
    mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    return mol

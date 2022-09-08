from pathlib import Path
import prody
import numpy as np
import rdkit
import torch

def parse_protein(pdb_id: str, data_dir: Path) -> prody.AtomGroup:
    """Convert pdb file to prody AtomGroup object.

    Args:
        pdb_id (str): PDB ID
        data_dir (Path): Filepath of PDBbind data

    Returns:
        prody.AtomGroup: All of the atoms in the pdb file.
    """
    pdb_path = data_dir / pdb_id / f'{pdb_id}_protein.pdb'
    pdb_path = str(pdb_path)
    protein_atoms = prody.parsePDB(pdb_path)
    return protein_atoms


def parse_ligand(pdb_id: str, data_dir: Path):
    """Load ligand file into rdkit, retrieve atom features and positions.

    Args:
        pdb_id (str): PDB ID
        data_dir (Path): Filepath of PDBbind data

    Returns:
        ligand: rdkit molecule of the ligand
        atom_positions: (N, 3) torch tensor with coordinates of all ligand atoms
        atom_types: tbd
        atom_charges: tbd
    """

    # construct path to ligand file
    ligand_path = data_dir / pdb_id / f'{pdb_id}_ligand.sdf'

    # read ligand into a rdkit mol
    suppl = rdkit.Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=False)
    ligands = list(suppl)
    if len(ligands) > 1:
        raise NotImplementedError('Multiple ligands found. Code is not written to handle multiple ligands.')
    ligand = ligands[0]

    # get atom positions
    ligand_conformer = ligand.GetConformer()
    atom_positions = ligand_conformer.GetPositions()

    # get atom types and charges
    atom_types = []
    atom_charges = []
    for atom in ligand.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        atom_charges.append(atom.GetFormalCharge()) # equibind code calls ComputeGasteigerCharges(mol), not sure why/if necessary

    # convert numpy arrays to torch tensors
    # TODO: one-hot encode atom types here
    atom_positions = torch.tensor(atom_positions)
    atom_types = torch.tensor(atom_types)
    atom_charges = torch.tensor(atom_charges)
    
    return ligand, atom_positions, atom_types, atom_charges

def get_pocket_atoms(pdb_atoms: prody.AtomGroup, ligand_atom_positions, pocket_cutoff):
    # note that pocket_cutoff is in units of angstroms

    # TODO: maybe it would be better to find all atoms that are within some distance of any ligand atom
    # this is more computataionally expensive but might give more accurate models

    # get bounding box of ligand
    # get padded bounding box
    # get coordinates of all protein atoms
    # get atom types of all protein atoms
    # find all atoms in padded bounding box
    # find atoms that come within a threshold distance from a ligand atom
    # get interface points

    # get ligand center of mass
    ligand_com = ligand_atom_positions.mean(axis=0, keepdims=False)
    ligand_com = np.array(ligand_com, dtype=float)

    # select atoms within cutoff of ligand
    pocket_atoms = pdb_atoms.select(f'protein and within {pocket_cutoff} of center', center=ligand_com)
    pocket_atom_positions = pocket_atoms.getCoords()
    pocket_atom_types = pocket_atoms.getElements()
    pocket_atom_charges = pocket_atoms.getCharges()

    # convert numpy arrays to torch tensors
    pocket_atom_positions = torch.tensor(pocket_atom_positions)
    
    # TODO: one-hot encode atom types

    return pocket_atom_positions, pocket_atom_types, pocket_atom_charges
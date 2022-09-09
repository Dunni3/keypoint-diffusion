from pathlib import Path
import prody
import numpy as np
import rdkit
import torch
from scipy import spatial as spa

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
    atom_positions = torch.tensor(atom_positions)

    # get atom features
    atom_features = lig_atom_featurizer(ligand)
    
    return ligand, atom_positions, atom_features

def get_pocket_atoms(rec_atoms: prody.AtomGroup, ligand_atom_positions, box_padding, pocket_cutoff):
    # note that pocket_cutoff is in units of angstroms

    # get bounding box of ligand
    lower_corner = ligand_atom_positions.min(axis=0, keepdim=True).values
    upper_corner = ligand_atom_positions.max(axis=0, keepdim=True).values

    # get padded bounding box
    lower_corner -= box_padding
    upper_corner += box_padding

    # get coordinates and atom types of all protein atoms 
    # TODO: fix parse_protein so that it selects the *atoms we actually want* from the protein structure (waters? metals? etc.)... i.e., selectiion logic will be contained to parse_protein
    rec_atom_positions = rec_atoms.getCoords()
    rec_atom_features = rec_atom_featurizer(rec_atoms)

    # convert protein atom positions to pytorch tensor
    rec_atom_positions = torch.tensor(rec_atom_positions)

    # find all protein atoms in padded bounding box
    above_lower_corner = (rec_atom_positions >= lower_corner).all(axis=1)
    below_upper_corner = (rec_atom_positions <= upper_corner).all(axis=1)
    # bounding_box_mask is a boolean array with length equal to number of atoms in receptor structure, indicating whether each atom is in the "bounding box"
    bounding_box_mask = above_lower_corner & below_upper_corner 

    # get positions + features for bounding box atoms
    box_atom_positions = rec_atom_positions[bounding_box_mask]
    box_atom_features = rec_atom_features[bounding_box_mask]

    # find atoms that come within a threshold distance from a ligand atom
    all_distances = spa.distance_matrix(box_atom_positions, ligand_atom_positions)
    # NOTE: even though the argumenets to distance_matrix were pytorch tensors, the returned array is a numpy array
    min_dist_to_ligand = all_distances.min(axis=1) # distance from each box atom to closest ligand atom
    pocket_atom_mask = min_dist_to_ligand < pocket_cutoff
    pocket_atom_mask = torch.tensor(pocket_atom_mask)

    # get positions + features for pocket atoms
    rec_atom_positions = box_atom_positions[pocket_atom_mask]
    rec_atom_features = box_atom_features[pocket_atom_mask]

    # get interface points
    # TODO: apply clustering algorithm to summarize interface points
    # for now, the interface points will just be the binding pocket points
    # closest_ligand_index = all_distances.argmin(axis=1) # indicies of ligand atoms that are closest to each box atom

    return rec_atom_positions, rec_atom_features


def rec_atom_featurizer(rec_atoms: prody.AtomGroup):
    protein_atom_elements = rec_atoms.getElements()
    protein_atom_charges = rec_atoms.getCharges()
    # TODO: one-hot encode atom types
    return protein_atom_elements

def lig_atom_featurizer(ligand):
    atom_types = []
    atom_charges = []
    for atom in ligand.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        atom_charges.append(atom.GetFormalCharge()) # equibind code calls ComputeGasteigerCharges(mol), not sure why/if necessary

    # convert numpy arrays to torch tensors
    # TODO: one-hot encode atom types here
    # TODO: think about how we are returning atom features. the diffusion model formulation
    # from max welling's group requires that we treat integer and categorical variables separately
    atom_types = torch.tensor(atom_types)
    atom_charges = torch.tensor(atom_charges)
    return atom_types
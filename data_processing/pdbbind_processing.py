from pathlib import Path
import prody
import numpy as np
import rdkit
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import AllChem as Chem
import torch
from scipy import spatial as spa
import dgl

from typing import Iterable, Union, List, Dict

def parse_protein(pdb_path: Path) -> prody.AtomGroup:
    """Convert pdb file to prody AtomGroup object.

    Args:
        pdb_path (Path): Path to receptor pdb file. 

    Returns:
        prody.AtomGroup: All of the atoms in the pdb file.
    """
    protein_atoms = prody.parsePDB(str(pdb_path))
    return protein_atoms


def parse_ligand(pdb_id: str, data_dir: Path, element_map: Dict[str, int]):
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
    atom_features = lig_atom_featurizer(ligand, element_map)
    
    return ligand, atom_positions, atom_features

def get_pocket_atoms(rec_atoms: prody.AtomGroup, ligand_atom_positions: torch.Tensor, box_padding: Union[int, float], pocket_cutoff: Union[int, float], element_map: Dict[str, int]):
    # note that pocket_cutoff is in units of angstroms

    # get bounding box of ligand
    lower_corner = ligand_atom_positions.min(axis=0, keepdim=True).values
    upper_corner = ligand_atom_positions.max(axis=0, keepdim=True).values

    # get padded bounding box
    lower_corner -= box_padding
    upper_corner += box_padding

    # get positions, features, a residue indicies for all protein atoms
    # TODO: fix parse_protein so that it selects the *atoms we actually want* from the protein structure (waters? metals? etc.)... i.e., selectiion logic will be contained to parse_protein
    rec_atom_positions = rec_atoms.getCoords()
    rec_atom_features = rec_atom_featurizer(rec_atoms, element_map)
    rec_atom_residx = rec_atoms.getResindices()

    # convert protein atom positions to pytorch tensor
    rec_atom_positions = torch.tensor(rec_atom_positions)
    rec_atom_features = torch.tensor(rec_atom_features)

    # find all protein atoms in padded bounding box
    above_lower_corner = (rec_atom_positions >= lower_corner).all(axis=1)
    below_upper_corner = (rec_atom_positions <= upper_corner).all(axis=1)
    # bounding_box_mask is a boolean array with length equal to number of atoms in receptor structure, indicating whether each atom is in the "bounding box"
    bounding_box_mask = above_lower_corner & below_upper_corner 

    # get positions + residue indicies for bounding box atoms
    box_atom_positions = rec_atom_positions[bounding_box_mask]
    box_atom_residx = rec_atom_residx[bounding_box_mask]

    # find atoms that come within a threshold distance from a ligand atom
    all_distances = spa.distance_matrix(box_atom_positions, ligand_atom_positions)
    # NOTE: even though the argumenets to distance_matrix were pytorch tensors, the returned array is a numpy array
    min_dist_to_ligand = all_distances.min(axis=1) # distance from each box atom to closest ligand atom
    pocket_atom_mask = min_dist_to_ligand < pocket_cutoff
    pocket_atom_mask = torch.tensor(pocket_atom_mask)

    # get residue indicies of all pocket atoms
    pocket_atom_residx = box_atom_residx[pocket_atom_mask]

    # get a mask of all atoms having a residue index contained in pocket_atom_residx
    byres_pocket_atom_mask = np.isin(rec_atom_residx, pocket_atom_residx)

    # get positions + features for pocket atoms
    pocket_atom_positions = rec_atom_positions[byres_pocket_atom_mask]
    pocket_atom_features = rec_atom_features[byres_pocket_atom_mask, :]

    # get interface points
    # TODO: apply clustering algorithm to summarize interface points
    # for now, the interface points will just be the binding pocket points

    return pocket_atom_positions, pocket_atom_features, byres_pocket_atom_mask


def rec_atom_featurizer(rec_atoms: prody.AtomGroup, element_map: Dict[str, int]):
    protein_atom_elements: np.ndarray = rec_atoms.getElements()
    protein_atom_charges: np.ndarray = rec_atoms.getCharges().astype(int)
    # TODO: should atom charges be integers?

    # one-hot encode atom elements
    onehot_elements = onehot_encode_elements(protein_atom_elements, element_map)

    # concatenate atom elements and charges
    protein_atom_features = np.concatenate([onehot_elements, protein_atom_charges[:, None]], axis=1)

    return protein_atom_features

def lig_atom_featurizer(ligand, element_map: Dict[str, int]):
    atom_elements = []
    atom_charges = []
    # TODO: do i need to explicitly compute atomic charges?
    for atom in ligand.GetAtoms():
        atom_elements.append(atom.GetSymbol())
        atom_charges.append(atom.GetFormalCharge()) # equibind code calls ComputeGasteigerCharges(mol), not sure why/if necessary

    # convert numpy arrays to torch tensors
    # TODO: think about how we are returning atom features. the diffusion model formulation
    # from max welling's group requires that we treat integer and categorical variables separately

    # one-hot encode atom elements
    onehot_elements = onehot_encode_elements(atom_elements, element_map)

    # convert charges to numpy array
    atom_charges = np.asarray(atom_charges, dtype=int)

    # concatenate atom elements and charges
    atom_features = np.concatenate([onehot_elements, atom_charges[:, None]], axis=1)

    # TODO: determine datatype for torch tensors
    atom_features = torch.tensor(atom_features)

    return atom_features

def onehot_encode_elements(atom_elements: Iterable, element_map: Dict[str, int]) -> np.ndarray:

    def element_to_idx(element_str, element_map=element_map):
        try:
            return element_map[element_str]
        except KeyError:
            return element_map['other']

    element_idxs = np.fromiter((element_to_idx(element) for element in atom_elements), int)
    onehot_elements = np.zeros((element_idxs.size, len(element_map)))
    onehot_elements[np.arange(element_idxs.size), element_idxs] = 1

    return onehot_elements

def build_receptor_graph(atom_positions: torch.Tensor, atom_features: torch.Tensor, k: int, edge_algorithm: str) -> dgl.DGLGraph:
    g = dgl.knn_graph(atom_positions, k=k, algorithm=edge_algorithm, dist='euclidean')
    g.ndata['x_0'] = atom_positions
    g.ndata['h_0'] = atom_features
    return g

def get_ot_loss_weights(ligand: rdkit.Chem.rdchem.Mol, pdb_path: Path, pocket_atom_mask: torch.Tensor):
    
    # i tried to implement this using rdkit but it didn't work
    # i need to implement the FF calculations using openmm, but that will take time that I simply don't have
    raise NotImplementedError
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

class Unparsable(Exception):
    pass

def parse_protein(pdb_path: Path, remove_hydrogen=False) -> prody.AtomGroup:
    """Convert pdb file to prody AtomGroup object.

    Args:
        pdb_path (Path): Path to receptor pdb file. 

    Returns:
        prody.AtomGroup: All of the atoms in the pdb file.
    """
    pdb_atoms = prody.parsePDB(str(pdb_path))

    # on rare occasions, pdb_atoms can be None
    # here is one such example: /home/ian/projects/mol_diffusion/DiffSBDD/data/crossdocked_pocket10/RDM1_ARATH_7_163_0/2q3t_A_rec_2q3t_cps_lig_tt_docked_262_pocket10.pdb
    # for now, we will skip these
    if pdb_atoms is None:
        raise Unparsable

    # exclude certain entities
    if remove_hydrogen:
        selection_str = 'not water and not hydrogen'
    else:
        selection_str = 'not water'

    protein_atoms = pdb_atoms.select(selection_str)

    return protein_atoms


def parse_ligand(ligand_path: Path, element_map: Dict[str, int], remove_hydrogen=False, include_charges=False):
    """Load ligand file into rdkit, retrieve atom features and positions.

    Args:
        ligand_path (Path): Filepath of ligand SDF file

    Returns:
        ligand: rdkit molecule of the ligand
        atom_positions: (N, 3) torch tensor with coordinates of all ligand atoms
        atom_types: tbd
        atom_charges: tbd
    """
    # read ligand into a rdkit mol
    suppl = rdkit.Chem.SDMolSupplier(str(ligand_path), sanitize=False, removeHs=remove_hydrogen)
    ligands = list(suppl)
    if len(ligands) > 1:
        raise NotImplementedError('Multiple ligands found. Code is not written to handle multiple ligands.')
    ligand = ligands[0]

    # actually remove all hydrogens - setting removeHs=True still preserves hydrogens that are necessary for specifying stereochemistry
    # note that therefore, this step destroys stereochemistry
    if remove_hydrogen:
        ligand = Chem.RemoveAllHs(ligand, sanitize=False)

    # get atom positions
    ligand_conformer = ligand.GetConformer()
    atom_positions = ligand_conformer.GetPositions()
    atom_positions = torch.tensor(atom_positions).float()

    atom_features, other_atoms_mask = lig_atom_featurizer(element_map, ligand=ligand)

    # skip ligands which have "other" type atoms
    if other_atoms_mask.sum() > 0:
        raise Unparsable

    # drop the "other" atom dimension
    atom_features = atom_features[:, :-1]
    
    return ligand, atom_positions, atom_features

def get_pocket_atoms(rec_atoms: prody.Selection, ligand_atom_positions: torch.Tensor, box_padding: Union[int, float], 
        pocket_cutoff: Union[int, float], element_map: Dict[str, int],
        interface_distance_threshold: float,
        interface_exclusion_threshold: float):
    # note that pocket_cutoff is in units of angstroms

    # get bounding box of ligand
    lower_corner = ligand_atom_positions.min(axis=0, keepdim=True).values
    upper_corner = ligand_atom_positions.max(axis=0, keepdim=True).values

    # get padded bounding box
    lower_corner -= box_padding
    upper_corner += box_padding

    # get positions, features, and residue indicies for all protein atoms
    rec_atom_positions = rec_atoms.getCoords()
    rec_atom_features, other_atoms_mask = rec_atom_featurizer(element_map=element_map, rec_atoms=rec_atoms)
    rec_atom_residx = rec_atoms.getResindices()

    # convert protein atom positions to pytorch tensor
    rec_atom_positions = torch.tensor(rec_atom_positions).float()
    rec_atom_features = torch.tensor(rec_atom_features).float()

    # remove "other" atoms from the receptor
    rec_atom_positions = rec_atom_positions[~other_atoms_mask]
    rec_atom_features = rec_atom_features[~other_atoms_mask]
    rec_atom_residx = torch.tensor(rec_atom_residx, dtype=int)[~other_atoms_mask]

    # find all protein atoms in padded bounding box
    above_lower_corner = (rec_atom_positions >= lower_corner).all(dim=1)
    below_upper_corner = (rec_atom_positions <= upper_corner).all(dim=1)
    # bounding_box_mask is a boolean array with length equal to number of atoms in receptor structure, indicating whether each atom is in the "bounding box"
    bounding_box_mask = above_lower_corner & below_upper_corner 

    # get positions + residue indicies for bounding box atoms
    box_atom_positions = rec_atom_positions[bounding_box_mask]
    box_atom_residx = rec_atom_residx[bounding_box_mask]

    # find atoms that come within a threshold distance from a ligand atom
    all_distances = spa.distance.cdist(box_atom_positions, ligand_atom_positions)
    all_distances = torch.tensor(all_distances)
    # NOTE: even though the argumenets to distance_matrix were pytorch tensors, the returned array is a numpy array
    min_dist_to_ligand, _ = all_distances.min(dim=1) # distance from each box atom to closest ligand atom
    pocket_atom_mask = min_dist_to_ligand < pocket_cutoff

    # get residue indicies of all pocket atoms
    pocket_atom_residx = box_atom_residx[pocket_atom_mask]

    # get a mask of all atoms having a residue index contained in pocket_atom_residx
    byres_pocket_atom_mask = torch.isin(rec_atom_residx, pocket_atom_residx)

    # get positions + features for pocket atoms
    pocket_atom_positions = rec_atom_positions[byres_pocket_atom_mask]
    pocket_atom_features = rec_atom_features[byres_pocket_atom_mask, :]

    # get interface points
    try:
        interface_points = get_interface_points(ligand_atom_positions, box_atom_positions, 
                                                dist_mat=all_distances.T, 
                                                distance_threshold=interface_distance_threshold,
                                                exclusion_threshold=interface_exclusion_threshold)
    except Exception as e:
        raise InterfacePointException(e)

    return pocket_atom_positions, pocket_atom_features, byres_pocket_atom_mask, interface_points


def rec_atom_featurizer(element_map: Dict[str, int], rec_atoms: prody.AtomGroup = None, protein_atom_elements: np.ndarray = None):

    if rec_atoms is None and protein_atom_elements is None:
        raise ValueError
    
    if protein_atom_elements is None:
        protein_atom_elements: np.ndarray = rec_atoms.getElements()

    # one-hot encode atom elements
    onehot_elements = onehot_encode_elements(protein_atom_elements, element_map)

    # get a mask for atoms that have the "other" category
    other_atoms_mask = torch.tensor(onehot_elements[:, -1] == 1).bool()

    # remove "other" category from onehot_elements
    # note that here we are assuming that the other category is the last in the onehot encoding
    protein_atom_features = onehot_elements[:, :-1]

    return protein_atom_features, other_atoms_mask

def lig_atom_featurizer(element_map: Dict[str, int], ligand=None, atom_elements: List[str] = None):

    if ligand is None and atom_elements is None:
        raise ValueError
    
    if ligand is not None:
        atom_elements = []
        atom_charges = []
        # TODO: do i need to explicitly compute atomic charges?
        for atom in ligand.GetAtoms():
            atom_elements.append(atom.GetSymbol())

    # convert numpy arrays to torch tensors
    # TODO: think about how we are returning atom features. the diffusion model formulation
    # from max welling's group requires that we treat integer and categorical variables separately

    # one-hot encode atom elements
    onehot_elements = onehot_encode_elements(atom_elements, element_map) # boolean array of shape (n_atoms, n_atom_types)

    # get a mask for atoms that have the "other" category
    other_atoms_mask = torch.tensor(onehot_elements[:, -1] == 1).bool()

    # convert charges to torch tensor
    atom_features = onehot_elements
    atom_features = torch.tensor(atom_features).bool()

    return atom_features, other_atoms_mask

def onehot_encode_elements(atom_elements: Iterable, element_map: Dict[str, int]) -> np.ndarray:

    def element_to_idx(element_str, element_map=element_map):
        try:
            return element_map[element_str]
        except KeyError:
            # print(f'other element found: {element_str}')
            return element_map['other']

    element_idxs = np.fromiter((element_to_idx(element) for element in atom_elements), int)
    onehot_elements = np.zeros((element_idxs.size, len(element_map)))
    onehot_elements[np.arange(element_idxs.size), element_idxs] = 1

    return onehot_elements

def build_receptor_graph(atom_positions: torch.Tensor, atom_features: torch.Tensor, k: int, edge_algorithm: str) -> dgl.DGLGraph:
    g = dgl.knn_graph(atom_positions, k=k, algorithm=edge_algorithm, dist='euclidean', exclude_self=True)
    g.ndata['x_0'] = atom_positions
    g.ndata['h_0'] = atom_features
    return g

def get_ot_loss_weights(ligand: rdkit.Chem.rdchem.Mol, pdb_path: Path, pocket_atom_mask: torch.Tensor):
    
    # i tried to implement this using rdkit but it didn't work
    # i need to implement the FF calculations using openmm, but that will take time that I simply don't have
    raise NotImplementedError

def center_complex(receptor_graph: dgl.DGLGraph, ligand_atom_positions: torch.Tensor):

    
    lig_com = ligand_atom_positions.mean(dim=0, keepdim=True)
    
    receptor_graph.ndata["x_0"] = receptor_graph.ndata["x_0"] - lig_com

    new_lig_pos = ligand_atom_positions - lig_com

    return receptor_graph, new_lig_pos

def get_interface_points(ligand_positions: torch.Tensor, rec_positions: torch.Tensor,
                         dist_mat: torch.Tensor = None, 
                         distance_threshold: float = 5, 
                         exclusion_threshold: float = 2):
    
    if dist_mat is None:
        dist_mat = spa.distance.cdist(ligand_positions, rec_positions)
        dist_mat = torch.cdist(ligand_positions, rec_positions)

    assert dist_mat.shape[0] == ligand_positions.shape[0]

    lig_idx, rec_idx = torch.where(dist_mat < distance_threshold)

    interface_points = (ligand_positions[lig_idx] + rec_positions[rec_idx])/2

    # select a subset of interface points such that all points in the subset have pairwise distances below 
    # the exclusion_threshold
    selected_interface_idxs = [0]
    for idx in range(1, interface_points.shape[0]):
        selected_points = interface_points[selected_interface_idxs]
        if len(selected_points.shape) == 1:
            selected_points = selected_points[None, :]

        candidate_interface_point = interface_points[idx][None, :]
        dist_to_candidate = torch.cdist(candidate_interface_point, selected_points)
        if torch.all(dist_to_candidate >= exclusion_threshold):
            selected_interface_idxs.append(idx)

    interface_points = interface_points[selected_interface_idxs]

    return interface_points

class InterfacePointException(Exception):

    def __init__(self, original_exception: Exception, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_exception = original_exception


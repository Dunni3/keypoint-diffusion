import warnings
import tempfile

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from openbabel import openbabel
import io

from utils import write_xyz_file

# this file has been taken from / adapted from DiffSBDD's file of the same name

def build_molecule(positions, atom_elements, add_hydrogens=False, sanitize=False, relax_iter=0, largest_frag=False):
    """Build a molecule from 3D positions and atom elements.

    Args:
        positions (_type_): N x 3 array of atom positions
        atom_elements (_type_): List of length N containing the string of every atom's element
        add_hydrogens (bool, optional): Whether to add hydrogens to molecule. Defaults to False.
        sanitize (bool, optional): sanitize kwarg for rdkit's SDF reader. Defaults to False.
        relax_iter (int, optional): How many iterations to use during FF minimization. Defaults to 0.
        largest_frag (bool, optional): Whether to select/return only largest frag from molecule. Defaults to False.

    Returns:
        rdkit molecule: Processed molecule.
    """

    mol = make_mol_openbabel(positions, atom_elements)
    if mol is None:
        return mol
    processed_mol = process_molecule(mol, add_hydrogens=add_hydrogens, 
        sanitize=sanitize, relax_iter=relax_iter, largest_frag=largest_frag)
    
    return processed_mol

def make_mol_openbabel(positions, atom_elements):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_elemens: N, containing element string of each atom
    Returns:
        rdkit molecule
    """
    # Write xyz file
    xyz_str = write_xyz_file(positions, atom_elements)

    # Convert to sdf file with openbabel
    # openbabel will add bonds
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")
    ob_mol = openbabel.OBMol()
    obConversion.ReadString(ob_mol, xyz_str)

    # convert sdf to rdkit molecule
    mol = Chem.MolFromMolBlock(obConversion.WriteString(ob_mol))

    return mol


def process_molecule(rdmol, add_hydrogens=False, sanitize=False, relax_iter=0,
                     largest_frag=False):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def filter_rd_mol(rdmol):
    """
    Filter out RDMols if they have a 3-3 ring intersection
    adapted from:
    https://github.com/luost26/3D-Generative-SBDD/blob/main/utils/chem.py
    """
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:
            continue
        for j, ring_b in enumerate(rings):
            if i <= j:
                continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0): 
                return False

    return True

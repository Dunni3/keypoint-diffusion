from Bio.PDB import PDBParser, PDBIO, MMCIFIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBIO import Select
import numpy as np
from scipy.spatial.distance import cdist
import rdkit.Chem as Chem
from pathlib import Path

from data_processing.pdbbind_processing import Unparsable

class PocketSelector(Select):

    def __init__(self, residues: list):
        super().__init__()
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues

def write_pocket_file(rec_file: Path, lig_file: Path, output_pocket_file: Path, cutoff: float = 5):

    # parse pdb file
    pdb_struct = PDBParser(QUIET=True).get_structure('', rec_file)

    # get ligand positions
    ligand = Chem.MolFromMolFile(str(lig_file), sanitize=False)
    if ligand is None:
        raise Unparsable(f'ligand {lig_file} could not be parsed')
    ligand_conformer = ligand.GetConformer()
    atom_positions = ligand_conformer.GetPositions()

    # get binding pocket residues
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        if not is_aa(residue.get_resname(), standard=True):
            continue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        is_pocket_residue = cdist(atom_positions, res_coords).min() < cutoff
        if is_pocket_residue:
            pocket_residues.append(residue)

    # save just the pocket residues
    pocket_selector = PocketSelector(pocket_residues)
    pdb_io = PDBIO()
    pdb_io.set_structure(pdb_struct)
    pdb_io.save(str(output_pocket_file), pocket_selector)

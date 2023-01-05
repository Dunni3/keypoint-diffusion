from pathlib import Path
import torch
import pickle
from typing import List
from rdkit import Chem
from torch.nn.functional import one_hot

from models.ligand_diffuser import LigandDiffuser
from data_processing.crossdocked.dataset import CrossDockedDataset
from analysis.molecule_builder import make_mol_openbabel

class ModelAnalyzer:

    def __init__(self, model: LigandDiffuser, dataset: CrossDockedDataset, device):
        self.model = model
        self.dataset = dataset
        self.connectivity_thresh = 0.5

        self.device = device

        # create ligand atom type distribution object
        type_counts_file = dataset.dataset_dir / 'train_type_counts.pkl'
        self.lig_type_dist = LigandTypeDistribution(type_counts_file)

        # open the set of training data smiles strings
        train_smiles_file = dataset.dataset_dir / 'train_smiles.pkl'
        with open(train_smiles_file, 'rb') as f:
            self.train_smiles: set = pickle.load(f)

    @torch.no_grad()
    def sample_and_analyze(self, n_receptors: int = 10, n_replicates: int = 10, rec_enc_batch_size: int = 16, diff_batch_size: int = 32):

        # randomly select n_receptors from the dataset
        receptor_idxs = torch.randint(low=0, high=len(self.dataset), size=(n_receptors,))
        rec_graphs = [self.dataset[int(idx)][0].to(device=self.device) for idx in receptor_idxs]

        # sample n_replicates ligands in each receptor
        samples = self.model.sample_random_sizes(
            rec_graphs, 
            n_replicates=n_replicates, 
            rec_enc_batch_size=rec_enc_batch_size, 
            diff_batch_size=diff_batch_size)

        # flatten the list of samples and separate into "positions" and "features"
        lig_pos = []
        lig_feat = []
        for rec_dict in samples:
            lig_pos.extend(rec_dict['positions'])
            lig_feat.extend(rec_dict['features'])

        # compute KL divergence between atom types in this sample vs. the training dataset
        atom_type_kldiv = self.lig_type_dist.kl_divergence(lig_feat)

        # convert to molecules
        unprocessed_mols = []
        for lig_pos_i, lig_feat_i in zip(lig_pos, lig_feat):
            element_idxs = torch.argmax(lig_feat_i, dim=1).tolist()
            atom_elements = self.dataset.lig_atom_idx_to_element(element_idxs)
            mol = make_mol_openbabel(lig_pos_i, atom_elements)
            unprocessed_mols.append(mol)

        # compute connectivity, validity, uniqueness, and novelty
        connected_mols, frac_connected = self.compute_connectivity(unprocessed_mols)
        valid_mol_smiles, frac_valid = self.compute_validity(connected_mols)
        unique_smiles, frac_unique = self.compute_uniqueness(valid_mol_smiles)
        novel_smiles, frac_novel = self.compute_novelty(unique_smiles)

        metrics = dict(
            atom_type_kldiv=atom_type_kldiv, 
            frac_valid=frac_valid, 
            frac_unique=frac_unique, 
            frac_novel=frac_novel)

        return metrics

    def compute_connectivity(self, mols):
        if len(mols) == 0:
            return [], 0.0

        # compute frac connected
        connected_mols = []
        for mol in mols:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = \
                max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                connected_mols.append(largest_mol)

        return connected_mols, len(connected_mols)/len(mols)

    def compute_validity(self, mols):
        if len(mols) == 0:
            return [], 0.0

        # compute validity
        valid_mol_smiles = []
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue
            smiles = Chem.MolToSmiles(mol)
            if smiles is not None:
                valid_mol_smiles.append(mol)

        return valid_mol_smiles, len(valid_mol_smiles)/len(mols)


    def compute_uniqueness(self, smiles: List[str]):
        if len(smiles) == 0:
            return [], 0.0

        unique_smiles = list(set(smiles))
        return unique_smiles, len(unique_smiles) / len(smiles)

    def compute_novelty(self, smiles: List[str]):
        if len(smiles) == 0:
            return [], 0.0

        novel_smiles = [ smi for smi in smiles if smi in self.train_smiles ]
        return novel_smiles, len(novel_smiles) / len(smiles)

# adapted from DiffSBDD's CategoricalDistribution class in their Metrics file
class LigandTypeDistribution:

    EPS = 1e-10

    def __init__(self, type_counts_file: Path):
        # get the counts of each atom type in the dataset
        with open(type_counts_file, 'rb') as f:
            type_counts: torch.Tensor = pickle.load(f)

        # note that type_counts is a 1-dim vector of length == number of atom types
        # each value of type_counts is the number of atoms in the dataset having a particular type
        # next, we convert these counts to probabilties
        self.p = type_counts / type_counts.sum()

    def kl_divergence(self, sample_atom_types: List[torch.Tensor]):

        sample_concat = torch.concat(sample_atom_types, dim=0)
        sample_onehot = one_hot(sample_concat.argmax(dim=1))

        sample_counts = sample_onehot.sum(dim=0)
        q = sample_counts / sample_counts.sum()
        q = q.to(self.p.device)

        kl_div = -torch.sum(self.p* torch.log(q / self.p + self.EPS ))

        return float(kl_div)
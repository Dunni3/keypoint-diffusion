from pathlib import Path
import torch
import pickle
from typing import List
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from analysis.SA_Score.sascorer import calculateScore
from torch.nn.functional import one_hot
import time

from models.ligand_diffuser import LigandDiffuser
from data_processing.crossdocked.dataset import CrossDockedDataset
from analysis.molecule_builder import make_mol_openbabel
from constants import allowed_bonds

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
        sampling_start = time.time()
        samples = self.model.sample_random_sizes(
            rec_graphs, 
            n_replicates=n_replicates, 
            rec_enc_batch_size=rec_enc_batch_size, 
            diff_batch_size=diff_batch_size)
        sample_time = time.time() - sampling_start
        print(f'sampling {n_receptors=} and {n_replicates=}')
        print(f'sampling time per molecule = {sample_time/(n_receptors*n_replicates):.2f} s', flush=True)

        # flatten the list of samples into "positions" and "features"
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

        # get metrics that operate on imperfect molecules
        atom_validity = self.check_atom_valency(unprocessed_mols)
        avg_frag_frac = self.compute_avg_frag_size(unprocessed_mols)

        # compute connectivity, validity, uniqueness, and novelty
        valid_mols, validity = self.compute_validity(unprocessed_mols)
        connected_smiles, connectivity = self.compute_connectivity(valid_mols)
        unique_smiles, uniqueness = self.compute_uniqueness(connected_smiles)
        _, novelty = self.compute_novelty(unique_smiles)

        metrics = dict(
            atom_type_kldiv=atom_type_kldiv,
            atom_validity=atom_validity,
            avg_frag_frac=avg_frag_frac, 
            validity=validity,
            connectivity=connectivity, 
            uniqueness=uniqueness, 
            novelty=novelty)

        return metrics

    def compute_connectivity(self, mols):
        if len(mols) == 0:
            return [], 0.0

        # compute frac connected
        connected_smiles = []
        for mol in mols:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = \
                max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                smiles = Chem.MolToSmiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)

        return connected_smiles, len(connected_smiles)/len(mols)

    def compute_validity(self, mols):
        if len(mols) == 0:
            return [], 0.0

        # compute validity
        valid_mols = []
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue
            valid_mols.append(mol)

        return valid_mols, len(valid_mols)/len(mols)


    def compute_uniqueness(self, smiles: List[str]):
        if len(smiles) == 0:
            return [], 0.0

        unique_smiles = list(set(smiles))
        return unique_smiles, len(unique_smiles) / len(smiles)

    def compute_novelty(self, smiles: List[str]):
        if len(smiles) == 0:
            return [], 0.0

        novel_smiles = [ smi for smi in smiles if smi not in self.train_smiles ]
        return novel_smiles, len(novel_smiles) / len(smiles)

    def detect_chemistry_problems(self, mols):
        # note this method is presently unused

        for mol in mols:
            problems = Chem.DetectChemistryProblems(mol)
            print(problems)

    def check_atom_valency(self, mols) -> float:
        """Checks the valency of individual atoms and returns the fraction which have valid valencies.

        The valency of an atom is considered invalid if an atom's explicit valency is 0 or greater than
        the maximum allowable valency. For example, carbon can have 4 bonds. In practice, since we don't generate hydrogens,
        if a carbon atom has 2 bonds, this is plausible bc it could have 2 implicit hydrogens. But if a carbon atom has 0 or 5 explicit bonds,
        this is obviously wrong and so such atoms would be marked as "invalid".
        """

        n_invalid_atoms = 0
        n_atoms = 0
        for mol in mols:

            n_atoms += mol.GetNumAtoms()

            for atom in mol.GetAtoms():

                # get the atom element as a string and its explicit valence
                element: str = atom.GetSymbol()
                explicit_valence: int = atom.GetExplicitValence()

                # get the maximum number of allowable bonds for this element
                if isinstance(allowed_bonds[element], int):
                    max_bonds = allowed_bonds[element]
                else:
                    max_bonds = max(allowed_bonds[element])

                if explicit_valence == 0 or explicit_valence > max_bonds:
                    n_invalid_atoms += 1

        
        atom_validity = 1 - n_invalid_atoms/n_atoms
        return atom_validity

    def compute_avg_frag_size(self, mols) -> float:
        """Returns the average fraction of atoms that belong to the largest fragment of a molecule."""

        frag_fracs = []
        for mol in mols:
            # get fragments
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            # get largest fragment
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            frag_fracs.append(largest_mol.GetNumAtoms() / mol.GetNumAtoms())

        return sum(frag_fracs) / len(frag_fracs)
                


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
        sample_onehot = one_hot(sample_concat.argmax(dim=1), sample_concat.shape[1])

        sample_counts = sample_onehot.sum(dim=0)
        q = sample_counts / sample_counts.sum()
        q = q.to(self.p.device)

        kl_div = -torch.sum(self.p* torch.log(q / (self.p + self.EPS) + self.EPS ))

        return float(kl_div)

# this class is taken from DiffSBDD
class MoleculeProperties:

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def evaluate(self, pocket_rdmols):
        """
        Run full evaluation
        Args:
            pocket_rdmols: list of lists, the inner list contains all RDKit
                molecules generated for a pocket
        Returns:
            QED, SA, LogP, Lipinski (per molecule), and Diversity (per pocket)
        """

        for pocket in pocket_rdmols:
            for mol in pocket:
                Chem.SanitizeMol(mol)
                assert mol is not None, "only evaluate valid molecules"

        all_qed = []
        all_sa = []
        all_logp = []
        all_lipinski = []
        per_pocket_diversity = []
        for pocket in tqdm(pocket_rdmols):
            all_qed.append([self.calculate_qed(mol) for mol in pocket])
            all_sa.append([self.calculate_sa(mol) for mol in pocket])
            all_logp.append([self.calculate_logp(mol) for mol in pocket])
            all_lipinski.append([self.calculate_lipinski(mol) for mol in pocket])
            per_pocket_diversity.append(self.calculate_diversity(pocket))

        print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
              f"{len(pocket_rdmols)} pockets evaluated.")

        qed_flattened = [x for px in all_qed for x in px]
        print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")

        sa_flattened = [x for px in all_sa for x in px]
        print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")

        logp_flattened = [x for px in all_logp for x in px]
        print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")

        lipinski_flattened = [x for px in all_lipinski for x in px]
        print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")

        print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")

        return all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity
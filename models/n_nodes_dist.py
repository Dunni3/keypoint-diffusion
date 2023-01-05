from pathlib import Path
import pickle
from typing import Dict
import torch

class LigandSizeDistribution:

    def __init__(self, processed_dataset_dir: Path):

        # get the dictionary containing the counts of number of nodes
        lig_size_file = processed_dataset_dir / 'train_ligand_sizes.pkl'
        with open(lig_size_file, 'rb') as f:
            n_node_counts: Dict[int, int] = pickle.load(f)

        ligands_counted = sum( n_node_counts.values() )
        n_node_probs = { k:v/ligands_counted for k,v in n_node_counts.items() }

        idx_to_size, probs = list(zip(*n_node_probs.items()))
        self.idx_to_size = torch.Tensor(idx_to_size).int()
        probs = torch.Tensor(probs)

        self.dist = torch.distributions.Categorical(probs=probs)

    def sample(self, size) -> torch.Tensor:
        sampled_idxs = self.dist.sample(sample_shape=size)
        sampled_sizes = self.idx_to_size[sampled_idxs]
        return sampled_sizes
        
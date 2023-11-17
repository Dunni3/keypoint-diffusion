from pathlib import Path
import pickle
from typing import Dict
import torch

class LigandSizeDistribution:

    def __init__(self, processed_dataset_dir: Path):

        joint_dist_file = processed_dataset_dir / 'train_n_node_joint_dist.pkl'
        if not joint_dist_file.exists():
            raise ValueError(f'Joint distribution file {joint_dist_file} does not exist')
        
        with open(joint_dist_file, 'rb') as f:
            joint_histogram, rec_bounds, lig_bounds = pickle.load(f) 

        self.joint_histogram = torch.from_numpy(joint_histogram)
        self.rec_bounds = rec_bounds
        self.lig_bounds = lig_bounds

        self.rec_idx_to_size = torch.arange(rec_bounds[0], rec_bounds[1]+1)
        self.lig_idx_to_size = torch.arange(lig_bounds[0], lig_bounds[1]+1)

        self.rec_size_to_idx = { int(size):idx for idx, size in enumerate(self.rec_idx_to_size) }
        self.lig_size_to_idx = { int(size):idx for idx, size in enumerate(self.lig_idx_to_size) }


        # # get the dictionary containing the counts of number of nodes
        # lig_size_file = processed_dataset_dir / 'train_ligand_sizes.pkl'
        # with open(lig_size_file, 'rb') as f:
        #     n_node_counts: Dict[int, int] = pickle.load(f)

        # ligands_counted = sum( n_node_counts.values() )
        # n_node_probs = { k:v/ligands_counted for k,v in n_node_counts.items() }

        # idx_to_size, probs = list(zip(*n_node_probs.items()))
        # self.idx_to_size = torch.Tensor(idx_to_size).int()
        # probs = torch.Tensor(probs)

        # self.dist = torch.distributions.Categorical(probs=probs)

    def sample(self, n_nodes_rec: torch.Tensor, n_replicates) -> torch.Tensor:

        for idx in range(n_nodes_rec.shape[0]):
            if int(n_nodes_rec[idx]) not in self.rec_size_to_idx:

                if n_nodes_rec[idx] < self.rec_bounds[0]:
                    new_size = self.rec_bounds[0]
                elif n_nodes_rec[idx] > self.rec_bounds[1]:
                    new_size = self.rec_bounds[1]

                print(f'WARNING: Number of receptor nodes {n_nodes_rec[idx]} is not in the range {self.rec_bounds} from the training set')
                print(f'Sampling number of ligand nodes conditioning on receptor having {new_size} nodes')
                n_nodes_rec[idx] = new_size

        rec_idxs = [ self.rec_size_to_idx[int(size)] for size in n_nodes_rec ]
        rec_idxs = torch.tensor(rec_idxs)
        lig_idxs = torch.multinomial(self.joint_histogram[rec_idxs], n_replicates, replacement=True)
        lig_sizes = self.lig_idx_to_size[lig_idxs]

        return lig_sizes
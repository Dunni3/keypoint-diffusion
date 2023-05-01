from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

# TODO: with the current implementation of fake atoms, the code will not function properly if max_fake_atom_frac = 0

class CrossDockedDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, 
        processed_data_file: str,
        rec_elements: List[str],
        lig_elements: List[str],
        pocket_edge_algorithm: str = 'bruteforce-blas',
        lig_box_padding: Union[int, float] = 6,
        pocket_cutoff: Union[int, float] = 4,
        receptor_k: int = 3,
        load_data: bool = True,
        use_boltzmann_ot: bool = False, 
        max_fake_atom_frac: float = 0.0,
        **kwargs):

        self.max_fake_atom_frac = max_fake_atom_frac

        # if load_data is false, we don't want to actually process any data
        self.load_data = load_data

        # define filepath of data
        self.data_file: Path = Path(processed_data_file)

        # TODO: remove this line, this was only for debugging
        # data_split = self.data_file.stem
        # filenames_file = self.data_file.parent / f'{data_split}_filenames.pkl'
        # with open(filenames_file, 'rb') as f:
        #     self.filenames = pickle.load(f)
        ########

        # atom typing configurations
        self.rec_elements = rec_elements
        self.rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.rec_elements) }
        self.rec_element_map['other'] = len(self.rec_elements)

        self.lig_elements = lig_elements
        self.lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.lig_elements) }
        self.lig_element_map['other'] = len(self.lig_elements)

        self.lig_reverse_map = {v:k for k,v in self.lig_element_map.items()}

        # hyperparameters for protein graph
        self.receptor_k: int = receptor_k
        self.lig_box_padding: Union[int, float] = lig_box_padding
        self.pocket_cutoff: Union[int, float] = pocket_cutoff
        self.pocket_edge_algorithm: str = pocket_edge_algorithm
        self.use_boltzmann_ot = use_boltzmann_ot

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):
        rec_graph = self.data['receptor_graph'][i]
        lig_atom_positions: torch.Tensor = self.data['lig_atom_positions'][i]
        lig_atom_features: torch.Tensor = self.data['lig_atom_features'][i]

        # add fake atoms to ligand
        if self.max_fake_atom_frac > 0:

            n_real_atoms = lig_atom_positions.shape[0]

            # add extra column for "no atom" type to the atom features
            lig_atom_features = torch.concat([lig_atom_features, torch.zeros(n_real_atoms, 1, dtype=lig_atom_features.dtype)], dim=1)

            n_fake_max = math.ceil(self.max_fake_atom_frac*n_real_atoms) # maximum possible number of fake atoms
            n_fake = int(torch.randint(0, n_fake_max+1, (1,))) # sample number of fake atoms from Uniform(0, n_fake_max)

            # if we have decided to add a non-zero number of fake atoms
            if n_fake != 0:
                max_coords, _ = lig_atom_positions.max(dim=0, keepdim=True)
                min_coords, _ = lig_atom_positions.min(dim=0, keepdim=True)
                fake_atom_positions = torch.rand(n_fake, 3)*(max_coords - min_coords) + min_coords

                lig_atom_positions = torch.concat([lig_atom_positions, fake_atom_positions], dim=0)

                fake_atom_features = torch.zeros(n_fake, lig_atom_features.shape[1], dtype=lig_atom_features.dtype)
                fake_atom_features[:, -1] = 1

                lig_atom_features = torch.concat([lig_atom_features, fake_atom_features], dim=0)


        interface_points = self.data['interface_points'][i]
        return rec_graph, lig_atom_positions, lig_atom_features, interface_points

    def __len__(self):
        return len(self.data['receptor_graph'])

    def process(self):
        # load data into memory
        if not self.load_data:
            self.data = {'receptor_graph': []}
        else:
            with open(self.data_file, 'rb') as f:
                self.data = pickle.load(f)

    def lig_atom_idx_to_element(self, element_idxs: List[int]):
        atom_elements = [ self.lig_reverse_map[element_idx] for element_idx in element_idxs ]
        return atom_elements

    @property
    def type_counts_file(self) -> Path:
        dataset_split = self.data_file.name.split('_')[0]
        types_file = self.data_file.parent / f'{dataset_split}_type_counts.pkl'
        return types_file

    @property
    def dataset_dir(self) -> Path:
        return self.data_file.parent

    def get_files(self, idx: int):
        """Given an index of the dataset, return the filepath of the receptor pdb and ligand sdf."""

        return self.data['rec_files'][idx], self.data['lig_files'][idx]

        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    receptor_graphs, lig_atom_positions, lig_atom_features, interface_points = zip(*examples)

    # batch the receptor graphs together
    receptor_graphs = dgl.batch(receptor_graphs)
    return receptor_graphs, lig_atom_positions, lig_atom_features, interface_points

def get_dataloader(dataset: CrossDockedDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader

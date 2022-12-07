from pathlib import Path
from typing import Dict, List, Union

import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np
import prody
import rdkit
import torch
from scipy import spatial as spa

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

# TODO: figure out what ligand atom elements we would/should actually support. We don't really need to include the metals do we?

class CrossDockedDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, 
        processed_data_dir: str,
        rec_elements: List[str],
        lig_elements: List[str],
        pocket_edge_algorithm: str = 'bruteforce-blas',
        lig_box_padding: Union[int, float] = 6,
        pocket_cutoff: Union[int, float] = 4,
        receptor_k: int = 3,
        use_boltzmann_ot: bool = False, **kwargs):

        # define filepaths of data
        self.data_dir: Path = Path(processed_data_dir)

        # atom typing configurations
        self.rec_elements = rec_elements
        self.rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.rec_elements) }
        self.rec_element_map['other'] = len(self.rec_elements)

        self.lig_elements = lig_elements
        self.lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.lig_elements) }
        self.lig_element_map['other'] = len(self.lig_elements)

        # hyperparameters for protein graph
        self.receptor_k: int = receptor_k
        self.lig_box_padding: Union[int, float] = lig_box_padding
        self.pocket_cutoff: Union[int, float] = pocket_cutoff
        self.pocket_edge_algorithm: str = pocket_edge_algorithm

        self.use_boltzmann_ot = use_boltzmann_ot

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):
    
        # get filepaths of the processed data that we need
        example_dir = self.data_dir / str(i)
        rec_graph_path = example_dir / 'rec_graph.dgl'
        lig_atom_data_path = example_dir / 'ligand_data.pt'

        receptor_graph = dgl.load_graphs(str(rec_graph_path))[0][0]
        with open(lig_atom_data_path, 'rb') as f:
            lig_atom_data = torch.load(f)
        lig_atom_positions = lig_atom_data['lig_atom_positions']
        lig_atom_features = lig_atom_data['lig_atom_features']

        return receptor_graph, lig_atom_positions, lig_atom_features

    def __len__(self):
        return self._size

    def process(self):

        self._size = 0
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                raise ValueError(f'only directories should be in data_dir, but found {subdir}')

            self._size += 1

        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    receptor_graphs, lig_atom_positions, lig_atom_features = zip(*examples)

    # batch the receptor graphs together
    receptor_graphs = dgl.batch(receptor_graphs)
    return receptor_graphs, lig_atom_positions, lig_atom_features

def get_dataloader(dataset: CrossDockedDataset, batch_size: int, num_workers: int = 1) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

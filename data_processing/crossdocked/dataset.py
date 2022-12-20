from pathlib import Path
import pickle
from typing import Dict, List, Union

import dgl
from dgl.dataloading import GraphDataLoader
import torch

from data_processing.pdbbind_processing import (build_receptor_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

# TODO: figure out what ligand atom elements we would/should actually support. We don't really need to include the metals do we?

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
        use_boltzmann_ot: bool = False, **kwargs):

        # if load_data is false, we don't want to actually process any data
        self.load_data = load_data

        # define filepath of data
        self.data_file: Path = Path(processed_data_file)

        # TODO: remove this line, this was only for debugging
        data_split = self.data_file.stem
        filenames_file = self.data_file.parent / f'{data_split}_filenames.pkl'
        with open(filenames_file, 'rb') as f:
            self.filenames = pickle.load(f)
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
        data_dict = self.data[i]

        # TODO: remove this line, its for debugging only
        # rec_file = self.filenames['rec_files'][i]
        # lig_file = self.filenames['lig_files'][i]
        # print(f'{i=}, {rec_file=}, {lig_file=}', flush=True)

        return data_dict['receptor_graph'], data_dict['lig_atom_positions'], data_dict['lig_atom_features']

    def __len__(self):
        return len(self.data)

    def process(self):
        # load data into memory
        
        if not self.load_data:
            self.data = []
        else:
            with open(self.data_file, 'rb') as f:
                self.data = pickle.load(f)

    def lig_atom_idx_to_element(self, element_idxs: List[int]):
        atom_elements = [ self.lig_reverse_map[element_idx] for element_idx in element_idxs ]
        return atom_elements


        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    receptor_graphs, lig_atom_positions, lig_atom_features = zip(*examples)

    # batch the receptor graphs together
    receptor_graphs = dgl.batch(receptor_graphs)
    return receptor_graphs, lig_atom_positions, lig_atom_features

def get_dataloader(dataset: CrossDockedDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader

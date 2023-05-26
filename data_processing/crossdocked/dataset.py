from pathlib import Path
import pickle
from typing import Dict, List, Union
import math

import dgl
from dgl.dataloading import GraphDataLoader
import torch

from data_processing.pdbbind_processing import (build_initial_complex_graph,
                                                get_pocket_atoms, parse_ligand,
                                                parse_protein, get_ot_loss_weights)

# TODO: with the current implementation of fake atoms, the code will not function properly if max_fake_atom_frac = 0

class CrossDockedDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, 
        processed_data_file: str,
        rec_elements: List[str],
        lig_elements: List[str],
        n_keypoints: int,
        graph_cutoffs: dict,
        # pocket_edge_algorithm: str = 'bruteforce-blas',
        lig_box_padding: Union[int, float] = 6, # an argument that is only useful for processing crossdocked data, and it is never actually used by this class 
        pocket_cutoff: Union[int, float] = 4,
        receptor_k: int = 3,
        load_data: bool = True,
        use_boltzmann_ot: bool = False, 
        max_fake_atom_frac: float = 0.0,
        **kwargs):

        self.max_fake_atom_frac = max_fake_atom_frac
        self.n_keypoints = n_keypoints
        self.graph_cutoffs = graph_cutoffs

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

        # hyperparameters for protein graph. these are never actually used. but they're considered "Dataset" parameters i suppose??
        self.lig_box_padding: Union[int, float] = lig_box_padding
        self.pocket_cutoff: Union[int, float] = pocket_cutoff
        self.use_boltzmann_ot = use_boltzmann_ot

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):


        lig_start_idx, lig_end_idx = self.lig_segments[i:i+2]
        rec_start_idx, rec_end_idx = self.rec_segments[i:i+2]
        ip_start_idx, ip_end_idx = self.ip_segments[i:i+2]

        lig_pos = self.lig_pos[lig_start_idx:lig_end_idx]
        lig_feat = self.lig_feat[lig_start_idx:lig_end_idx]
        rec_pos = self.rec_pos[rec_start_idx:rec_end_idx]
        rec_feat = self.rec_feat[rec_start_idx:rec_end_idx]
        interface_points = self.interface_points[ip_start_idx:ip_end_idx]
        rec_res_idx = self.rec_res_idx[rec_start_idx:rec_end_idx]


        complex_graph = build_initial_complex_graph(rec_pos, rec_feat, rec_res_idx, n_keypoints=self.n_keypoints, cutoffs=self.graph_cutoffs, lig_atom_positions=lig_pos, lig_atom_features=lig_feat)

        # complex_graph = self.data['complex_graph'][i]
        # interface_points = self.data['interface_points'][i]

        # add fake atoms to ligand
        if self.max_fake_atom_frac > 0:

            n_real_atoms = lig_pos.shape[0]

            # add extra column for "no atom" type to the atom features
            lig_feat = torch.concat([lig_feat, torch.zeros(n_real_atoms, 1, dtype=lig_feat.dtype)], dim=1)

            n_fake_max = math.ceil(self.max_fake_atom_frac*n_real_atoms) # maximum possible number of fake atoms
            n_fake = int(torch.randint(0, n_fake_max+1, (1,))) # sample number of fake atoms from Uniform(0, n_fake_max)

            # if we have decided to add a non-zero number of fake atoms
            if n_fake != 0:
                max_coords, _ = lig_pos.max(dim=0, keepdim=True)
                min_coords, _ = lig_pos.min(dim=0, keepdim=True)
                fake_atom_positions = torch.rand(n_fake, 3)*(max_coords - min_coords) + min_coords

                lig_pos = torch.concat([lig_pos, fake_atom_positions], dim=0)

                fake_atom_features = torch.zeros(n_fake, lig_feat.shape[1], dtype=lig_feat.dtype)
                fake_atom_features[:, -1] = 1

                lig_feat = torch.concat([lig_feat, fake_atom_features], dim=0)

                complex_graph.add_nodes(n_fake, ntype='lig')
                complex_graph.nodes['lig'].data['x_0'] = lig_pos
                complex_graph.nodes['lig'].data['h_0'] = lig_feat
            else:
                complex_graph.nodes['lig'].data['h_0'] = lig_feat
        
        return complex_graph, interface_points

    def __len__(self):
        return self.lig_segments.shape[0] - 1

    def process(self):
        # load data into memory
        if not self.load_data:
            self.lig_segments = torch.tensor([0])
        else:

            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)

            self.lig_pos = data['lig_pos']
            self.lig_feat = data['lig_feat']
            self.rec_pos = data['rec_pos']
            self.rec_feat = data['rec_feat']
            self.interface_points = data['interface_points']
            self.rec_segments = data['rec_segments']
            self.lig_segments = data['lig_segments']
            self.ip_segments = data['ip_segments']
            self.rec_files = data['rec_files']
            self.lig_files = data['lig_files']
            self.rec_res_idx = data['rec_res_idx']

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

        return self.rec_files[idx], self.lig_files[idx]

        
def collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    complex_graphs, interface_points = zip(*examples)

    # batch the receptor graphs together
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, interface_points

def get_dataloader(dataset: CrossDockedDataset, batch_size: int, num_workers: int = 1, **kwargs) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return dataloader

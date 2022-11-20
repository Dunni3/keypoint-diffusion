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

# TODO: figure out what ligand atom elements we whould actually support. We don't really need to include the metals do we?

class PDBbind(dgl.data.DGLDataset):

    def __init__(self, name: str, index_file: str, 
        raw_data_dir: str,
        processed_data_dir: str,
        rec_elements: List[str],
        lig_elements: List[str],
        pocket_edge_algorithm: str = 'bruteforce-blas',
        lig_box_padding: Union[int, float] = 6,
        pocket_cutoff: Union[int, float] = 4,
        receptor_k: int = 3,
        ligand_k: int = 3,
        dataset_size: int = None,
        use_boltzmann_ot: bool = False):
        
        self.dataset_size: int = dataset_size

        # define filepaths of data
        self.index_file: Path = Path(index_file)
        self.raw_data_dir: Path = Path(raw_data_dir)
        self.processed_data_dir: Path = Path(processed_data_dir)

        # atom typing configurations
        self.rec_elements = rec_elements
        self.rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.rec_elements) }
        self.rec_element_map['other'] = len(self.rec_elements)

        self.lig_elements = lig_elements
        self.lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(self.lig_elements) }
        self.lig_element_map['other'] = len(self.lig_elements)


        # hyperparameters for protein graph
        self.receptor_k: int = receptor_k
        self.ligand_k: int = ligand_k
        self.lig_box_padding: Union[int, float] = lig_box_padding
        self.pocket_cutoff: Union[int, float] = pocket_cutoff
        self.pocket_edge_algorithm = pocket_edge_algorithm

        self.use_boltzmann_ot = use_boltzmann_ot

        super().__init__(name=name) # this has to happen last because this will call self.process()

    def __getitem__(self, i):
        pdb_id = self.pdb_ids[i] # get PDB ID of ith dataset example
        
        # get filepaths of the processed data that we need
        rec_graph_path = self.processed_data_dir / pdb_id / f'{pdb_id}_rec_graph.dgl'
        lig_atom_data_path = self.processed_data_dir / pdb_id / f'{pdb_id}_ligand_data.pt'

        receptor_graph = dgl.load_graphs(str(rec_graph_path))[0][0]
        with open(lig_atom_data_path, 'rb') as f:
            lig_atom_data = torch.load(f)
        lig_atom_positions = lig_atom_data['lig_atom_positions']
        lig_atom_features = lig_atom_data['lig_atom_features']

        return receptor_graph, lig_atom_positions, lig_atom_features

    def __len__(self):
        return len(self.pdb_ids)

    def process(self):

        # TODO: implement logic that only makes the graph objects if they don't already exist
        # if output_files already exist and force_reprocess is not true, then skip this pdb_id

        # get pdb ids from index file
        with open(self.index_file, 'r') as f:
            self.pdb_ids = [line.strip() for line in f]

        if self.dataset_size is not None:
            self.pdb_ids = self.pdb_ids[:self.dataset_size]

        # we will want to do this paralellized over PDBs
        # but for now, a simple for loop will do
        for pdb_id in self.pdb_ids:

            # construct filepath of receptor pdb
            pdb_path: Path = self.raw_data_dir / pdb_id / f'{pdb_id}_protein_nowater.pdb'

            # get all atoms from pdb file
            pdb_atoms: prody.AtomGroup = parse_protein(pdb_path)

            # get rdkit molecule from ligand, as well as atom positions/features
            ligand_path = self.raw_data_dir / pdb_id / f'{pdb_id}_ligand.sdf'
            ligand, lig_atom_positions, lig_atom_features = parse_ligand(ligand_path, element_map=self.lig_element_map)

            # get all protein atoms that form the binding pocket
            pocket_atom_positions, pocket_atom_features, pocket_atom_mask \
                 = get_pocket_atoms(pdb_atoms, lig_atom_positions, 
                 box_padding=self.lig_box_padding, pocket_cutoff=self.pocket_cutoff, element_map=self.rec_element_map)

            # get boltzmann probabilities for OT loss
            # TODO: implement computing boltzmann probabilities for OT loss
            if self.use_boltzmann_ot:
                ot_loss_boltzmann_weights = get_ot_loss_weights(ligand, pdb_path, pocket_atom_mask, pocket_atom_positions)

            # build receptor graph
            receptor_graph = build_receptor_graph(pocket_atom_positions, pocket_atom_features, self.receptor_k, self.pocket_edge_algorithm)

            # define filepaths for saving processed data
            output_dir = self.processed_data_dir / pdb_id
            output_dir.mkdir(exist_ok=True)
            

            # save receptor graph
            receptor_graph_path = output_dir / f'{pdb_id}_rec_graph.dgl'
            dgl.save_graphs(str(receptor_graph_path), receptor_graph)

            # save ligand data
            ligand_data_path = output_dir / f'{pdb_id}_ligand_data.pt'
            payload = {'lig_atom_positions': lig_atom_positions, 'lig_atom_features': lig_atom_features}
            with open(ligand_data_path, 'wb') as f:
                torch.save(payload, f)

def pdbbind_collate_fn(examples: list):

    # break receptor graphs, ligand positions, and ligand features into separate lists
    receptor_graphs, lig_atom_positions, lig_atom_features = zip(*examples)

    # batch the receptor graphs together
    receptor_graphs = dgl.batch(receptor_graphs)
    return receptor_graphs, lig_atom_positions, lig_atom_features

def get_pdb_dataloader(dataset: PDBbind, batch_size: int, num_workers: int = 1) -> GraphDataLoader:

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=num_workers, collate_fn=pdbbind_collate_fn)
    return dataloader

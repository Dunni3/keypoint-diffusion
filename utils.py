import openbabel
# from rdkit.Chem import AllChem as Chem
# from rdkit.Chem import rdDetermineBonds
import tempfile
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import dgl

# this is taken from DiffSBDD, minor modification to return the file contents without writing to disk if filename=None 
def write_xyz_file(coords, atom_types, filename = None):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"

    if filename is None:
        return out
    
    with open(filename, 'w') as f:
        f.write(out)

# this is taken from DiffSBDD's make_mol_openbabel, i've adapted their method to do bond determination using rdkit
# useful blog post: https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html
# update, i wasn't able to make this work, so
# TODO: make this work
# def make_mol_rdkit(positions, atom_types, atom_decoder):
#     """
#     Build an RDKit molecule using openbabel for creating bonds
#     Args:
#         positions: N x 3
#         atom_types: N
#         atom_decoder: maps indices to atom types
#     Returns:
#         rdkit molecule
#     """
#     atom_types = [atom_decoder[x] for x in atom_types]

#     # Write xyz file
#     xyz_file_contents = write_xyz_file(positions, atom_types)

#     # get rdkit mol object from xyz file
#     raw_mol = Chem.MolFromXYZBlock(xyz_file_contents)

#     # find bonds, without determining bond-orders
#     conn_mol = Chem.Mol(raw_mol)
#     conn_mol = rdDetermineBonds.DetermineConnectivity(conn_mol)

#     # Convert to sdf file with openbabel
#     # openbabel will add bonds
#     obConversion = openbabel.OBConversion()
#     obConversion.SetInAndOutFormats("xyz", "sdf")
#     ob_mol = openbabel.OBMol()
#     obConversion.ReadFile(ob_mol, tmp_file)

#     obConversion.WriteFile(ob_mol, tmp_file)

#     # Read sdf file with RDKit
#     mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

#     return mol

# its kind of dumb to have this one-liner, but i save the model in multiple places thorughout the codebase so i thought i would centralize the 
# model saving code incase i want to change it later
def save_model(model, output_file: Path):
    torch.save(model.state_dict(), str(output_file))


def get_rec_atom_map(dataset_config: dict):
    # construct atom typing maps
    rec_elements = dataset_config['rec_elements']
    rec_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(rec_elements) }
    rec_element_map['other'] = len(rec_elements)

    lig_elements = dataset_config['lig_elements']
    lig_element_map: Dict[str, int] = { element: idx for idx, element in enumerate(lig_elements) }
    lig_element_map['other'] = len(lig_elements)
    return rec_element_map, lig_element_map


def concat_graph_data(graph_data: List[torch.tensor], device=None):

    if device is None:
        device = graph_data[0].device

    concat_data = torch.concatenate(graph_data, dim=0)
    graph_sizes = torch.tensor([ x.shape[0] for x in graph_data], dtype=int, device=device)
    graph_idx = torch.arange(graph_sizes.shape[0], device=device).repeat_interleave(graph_sizes)
    graph_indptr = torch.zeros(len(graph_data)+1, device=device, dtype=int)
    graph_indptr[1:] = torch.cumsum(graph_sizes, dim=0)
    return concat_data, graph_idx, graph_indptr

def get_batch_info(g: dgl.DGLHeteroGraph) -> Tuple[dict,dict]:
    batch_num_nodes = {}
    for ntype in g.ntypes:
        batch_num_nodes[ntype] = g.batch_num_nodes(ntype)

    batch_num_edges = {}
    for etype in g.canonical_etypes:
        batch_num_edges[etype] = g.batch_num_edges(etype)

    return batch_num_nodes, batch_num_edges
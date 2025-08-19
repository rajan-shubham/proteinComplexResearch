"""
deeprank_gnn_preprocess.py

A practical, runnable Python module to perform Stage A of DeepRank-GNN:
- Parse PDB
- Extract interface residues between two chains by cutoff
- Compute node features (one-hot aa, centroid coords, simple charge, hydrophobicity,
  optional conservation if provided, SASA via Shrake-Rupley)
- Build residue-level graph (edges by CA distance cutoff)
- Export as PyTorch Geometric Data objects (or NetworkX)

Requirements:
  pip install biopython numpy pandas torch networkx torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
  (For PyG installation follow official instructions matching your CUDA/PyTorch.)

Notes:
- DSSP is NOT required here (we use Shrake-Rupley for SASA). If you prefer DSSP, add that as an option.
- Conservation scores are optional: pass a dict mapping (chain, resseq) -> score or a file.

Usage example at bottom.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import networkx as nx
import torch
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB import ShrakeRupley
import os
from Bio.PDB.Polypeptide import protein_letters_3to1
import matplotlib.pyplot as plt

def three_to_one(resname: str) -> str:
    return protein_letters_3to1.get(resname.capitalize(), "X")


# ----------------------------- Helper constants -----------------------------
AA_LIST = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
AA_TO_ONE = {a: a for a in AA_LIST}
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# Kyte-Doolittle hydrophobicity scale (per-residue)
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Simple net side-chain charge approximation at physiological pH
CHARGE_SIMPLE = {
    'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0, 'Q': 0.0, 'E': -1.0,
    'G': 0.0, 'H': 0.1, 'I': 0.0, 'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0,
    'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0
}

# ----------------------------- PDB parsing & interface -----------------------------

def load_structure(pdb_path: str, model_id: int = 0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    return structure[model_id]


def get_chain(structure, chain_id: str):
    for ch in structure:
        if ch.id == chain_id:
            return ch
    raise KeyError(f"Chain {chain_id} not found")


def extract_interface_residues(structure, chain_id_a: str, chain_id_b: str, cutoff: float = 8.5) -> List[Tuple[str,int]]:
    """
    Return list of interface residues as tuples (chain_id, resid) where resid is the residue id.number
    Uses any-atom distance cutoff.
    """
    chain_a = get_chain(structure, chain_id_a)
    chain_b = get_chain(structure, chain_id_b)

    atoms_a = Selection.unfold_entities(chain_a, 'A')  # atom list
    atoms_b = Selection.unfold_entities(chain_b, 'A')
    ns = NeighborSearch(atoms_b)

    interface_residues = set()

    for res in chain_a:
        # skip hetero or missing
        if res.id[0] != ' ':
            continue
        for atom in res:
            close = ns.search(atom.coord, cutoff)
            if len(close) > 0:
                interface_residues.add((chain_id_a, res.id[1]))
                break

    # repeat the other way
    ns2 = NeighborSearch(atoms_a)
    for res in chain_b:
        if res.id[0] != ' ':
            continue
        for atom in res:
            close = ns2.search(atom.coord, cutoff)
            if len(close) > 0:
                interface_residues.add((chain_id_b, res.id[1]))
                break

    # return sorted list for deterministic ordering
    return sorted(list(interface_residues), key=lambda x: (x[0], x[1]))

# ----------------------------- Features computation -----------------------------

def residue_one_hot(res):
    """Convert a residue object or 1-letter code to a 20-dim one-hot vector."""
    try:
        three = res.get_resname()
    except Exception:
        three = res
    try:
        one = three_to_one(three)
    except Exception:
        one = 'X'
    vec = np.zeros(len(AA_LIST), dtype=float)
    if one in AA_TO_INDEX:
        vec[AA_TO_INDEX[one]] = 1.0
    return vec, one


def residue_centroid(res):
    coords = [atom.coord for atom in res if atom.element != 'H']
    if len(coords) == 0:
        coords = [atom.coord for atom in res]
    return np.mean(coords, axis=0)


def compute_sasa_for_structure(structure):
    """Compute SASA per residue using Shrake-Rupley (returns dict (chain, resid)->sasa)
    Requires only Biopython.
    """
    sr = ShrakeRupley()
    sr.compute(structure, level='R')  # per residue
    sasa = {}
    for model in structure.get_parent():
        pass
    # structure here is a Model (because load_structure returns Model). ShrakeRupley modifies residues.
    for chain in structure:
        for res in chain:
            if res.id[0] != ' ':
                continue
            key = (chain.id, res.id[1])
            val = res.sasa if hasattr(res, 'sasa') else 0.0
            sasa[key] = float(val)
    return sasa


def compute_node_features(structure, interface_residues: List[Tuple[str,int]], conservation: Optional[Dict[Tuple[str,int], float]] = None) -> Tuple[List[Tuple[str,int]], np.ndarray]:
    """Return ordered list of residues and an (N, F) numpy array of features.
    Features included:
      - one-hot (20)
      - centroid coords (3)
      - charge (1)
      - hydrophobicity (1)
      - conservation (1) [optional]
      - sasa (1)
    """
    # compute sasa once
    sasa_map = compute_sasa_for_structure(structure)

    rows = []
    feats = []

    for (chain_id, resseq) in interface_residues:
        chain = get_chain(structure, chain_id)
        res = None
        for r in chain:
            if r.id[1] == resseq and r.id[0] == ' ':
                res = r
                break
        if res is None:
            raise KeyError(f"Residue {chain_id} {resseq} not found")

        one_hot, one_letter = residue_one_hot(res)
        centroid = residue_centroid(res)
        charge = CHARGE_SIMPLE.get(one_letter, 0.0)
        hyd = KD_SCALE.get(one_letter, 0.0)
        cons = 0.0
        if conservation and (chain_id, resseq) in conservation:
            cons = float(conservation[(chain_id, resseq)])
        sasa = sasa_map.get((chain_id, resseq), 0.0)

        feat = np.concatenate([one_hot, centroid, np.array([charge, hyd, cons, sasa])])
        rows.append((chain_id, resseq))
        feats.append(feat.astype(float))

    X = np.vstack(feats) if len(feats) > 0 else np.zeros((0, len(AA_LIST)+3+4))
    return rows, X

# ----------------------------- Graph construction -----------------------------

def build_residue_graph(nodes: List[Tuple[str,int]], node_coords: np.ndarray, cutoff: float = 8.0) -> nx.Graph:
    """Build an undirected NetworkX graph where nodes are indexed 0..N-1 corresponding to nodes list.
    Edges added if CA (or centroid) distance <= cutoff. Edge attributes include distance and unit vector.
    """
    G = nx.Graph()
    N = len(nodes)
    for i in range(N):
        G.add_node(i, chain=nodes[i][0], resseq=nodes[i][1])

    # compute pairwise distances
    coords = node_coords[:, :3]  # centroid
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    for i in range(N):
        for j in range(i+1, N):
            dij = dists[i, j]
            if dij <= cutoff:
                vec = coords[j] - coords[i]
                norm = np.linalg.norm(vec)
                unit = vec / norm if norm > 0 else np.zeros(3)
                G.add_edge(i, j, distance=float(dij), unit_vec=unit.tolist())
    return G


def networkx_to_pyg_data(G: nx.Graph, node_features: np.ndarray, edge_feature_names: Optional[List[str]] = None):
    """Convert networkx graph to a torch_geometric Data-like dict.
    Returns:
      - x: torch.Tensor [N, F]
      - edge_index: torch.LongTensor [2, E]
      - edge_attr: torch.Tensor [E, Fe]
      - metadata lists for nodes
    """
    import torch
    N = G.number_of_nodes()
    node_order = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_order)}

    edges = []
    edge_attrs = []
    for u, v, data in G.edges(data=True):
        edges.append([node_map[u], node_map[v]])
        # pack distance and unit vector as edge attr
        ea = [data.get('distance', 0.0)] + data.get('unit_vec', [0.0,0.0,0.0])
        edge_attrs.append(ea)

    if len(edges) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # For undirected graphs, duplicate edges both ways for PyG
        edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
        ea = np.array(edge_attrs, dtype=float)
        ea = np.vstack([ea, ea])  # duplicate
        edge_attr = torch.tensor(ea, dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)

    data = {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_metadata': [(G.nodes[n].get('chain'), G.nodes[n].get('resseq')) for n in node_order]
    }
    return data

# ----------------------------- High level pipeline -----------------------------

def preprocess_pdb_to_pyg(pdb_path: str, chain_a: str, chain_b: str, interface_cutoff: float = 8.5, graph_cutoff: float = 8.0, conservation: Optional[Dict[Tuple[str,int],float]] = None):
    """Full pipeline: load structure -> extract interface -> compute features -> build graph -> to PyG data dict"""
    model = load_structure(pdb_path)
    interface = extract_interface_residues(model, chain_a, chain_b, cutoff=interface_cutoff)
    if len(interface) == 0:
        raise ValueError("No interface residues found. Check chain ids or cutoff.")

    nodes, X = compute_node_features(model, interface, conservation=conservation)
    # node_coords are stored in columns [20..22] of features (because one-hot 20 + centroid 3)
    G = build_residue_graph(nodes, X, cutoff=graph_cutoff)
    pyg = networkx_to_pyg_data(G, X)
    return pyg

# ----------------------------- Example usage -----------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess PDB to graph for DeepRank-GNN')
    parser.add_argument('pdb', help='path to pdb file')
    parser.add_argument('--chainA', required=True, help='Chain A id')
    parser.add_argument('--chainB', required=True, help='Chain B id')
    parser.add_argument('--iface_cutoff', type=float, default=8.5)
    parser.add_argument('--graph_cutoff', type=float, default=8.0)
    parser.add_argument('--out', help='output .pt file path', default='graph_data.pt')
    args = parser.parse_args()

    data = preprocess_pdb_to_pyg(args.pdb, args.chainA, args.chainB, interface_cutoff=args.iface_cutoff, graph_cutoff=args.graph_cutoff)

    # save as torch file for PyG training
    try:
        torch.save(data, args.out)
        print('Saved PyG-like dict to', args.out)
        # Load your saved graph
        data = torch.load("graphs")

        edge_index = data["edge_index"]
        node_metadata = data["node_metadata"]

        # Build graph from edge_index
        G = nx.Graph()
        for src, dst in edge_index.T.tolist():
            G.add_edge(src, dst)

        # Assign colors based on chain (A = blue, B = red, unknown = gray)
        colors = []
        for i in range(len(G.nodes)):
            chain_id, resseq = node_metadata[i]
            if chain_id == "A":
                colors.append("blue")
            elif chain_id == "B":
                colors.append("red")
            else:
                colors.append("gray")

        # Node sizes based on degree (more connected = bigger)
        sizes = [2 + 5*G.degree(n) for n in G.nodes]

        # Draw the graph
        plt.figure(figsize=(8,8))
        nx.draw(
            G, 
            node_color=colors, 
            node_size=sizes, 
            with_labels=False, 
            edge_color="black"
        )
        plt.title("Protein Interface Graph (Chain A = Blue, Chain B = Red)")
        plt.show()
    except Exception as e:
        print('Could not save with torch (is it installed?). Data keys:', list(data.keys()))
        raise
# python3 deeprankGNNpreprocess/main.py deeprankGNNpreprocess/1a3n.pdb --chainA A --chainB C --out ./graphs
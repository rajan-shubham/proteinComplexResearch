import json
import os
import requests
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
import pickle

# -------------------------
# Amino acid properties
# -------------------------
AMINO_ACID_PROPERTIES = {
    'ALA': {'hydrophobicity': 1.8, 'mw': 89.09},
    'ARG': {'hydrophobicity': -4.5, 'mw': 174.20},
    'ASN': {'hydrophobicity': -3.5, 'mw': 132.12},
    'ASP': {'hydrophobicity': -3.5, 'mw': 133.10},
    'CYS': {'hydrophobicity': 2.5, 'mw': 121.16},
    'GLN': {'hydrophobicity': -3.5, 'mw': 146.15},
    'GLU': {'hydrophobicity': -3.5, 'mw': 147.13},
    'GLY': {'hydrophobicity': -0.4, 'mw': 75.07},
    'HIS': {'hydrophobicity': -3.2, 'mw': 155.16},
    'ILE': {'hydrophobicity': 4.5, 'mw': 131.18},
    'LEU': {'hydrophobicity': 3.8, 'mw': 131.18},
    'LYS': {'hydrophobicity': -3.9, 'mw': 146.19},
    'MET': {'hydrophobicity': 1.9, 'mw': 149.21},
    'PHE': {'hydrophobicity': 2.8, 'mw': 165.19},
    'PRO': {'hydrophobicity': -1.6, 'mw': 115.13},
    'SER': {'hydrophobicity': -0.8, 'mw': 105.09},
    'THR': {'hydrophobicity': -0.7, 'mw': 119.12},
    'TRP': {'hydrophobicity': -0.9, 'mw': 204.23},
    'TYR': {'hydrophobicity': -1.3, 'mw': 181.19},
    'VAL': {'hydrophobicity': 4.2, 'mw': 117.15},
    'HOH': {'hydrophobicity': -0.4, 'mw': 18.02},
    'UNK': {'hydrophobicity': 0.0, 'mw': 0.0}
}

# -------------------------
# PDB download
# -------------------------
alphafold_api_url = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
output_dir = "pdb_files"
os.makedirs(output_dir, exist_ok=True)

def fetch_pdb(uniprot_id):
    url = alphafold_api_url.format(uniprot_id=uniprot_id)
    response = requests.get(url)
    if response.status_code == 200:
        pdb_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(response.text)
        print(f"✅ Saved {uniprot_id} → {pdb_path}")
        return pdb_path
    else:
        print(f"❌ Failed to fetch {uniprot_id} (HTTP {response.status_code})")
        return None

# -------------------------
# PDB → Graph Data
# -------------------------
parser = PDBParser(QUIET=True)

def pdb_to_data(pdb_path, label=0):
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)

    ca_atoms, residues, chain_ids = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"])
                    residues.append(residue)
                    chain_ids.append(chain.id)

    n = len(ca_atoms)
    if n == 0:
        return None

    coords = np.vstack([atom.coord for atom in ca_atoms])

    # residue names + frequency
    res_names = [res.get_resname() if res.get_resname() in AMINO_ACID_PROPERTIES else 'UNK'
                 for res in residues]
    unique, counts = np.unique(res_names, return_counts=True)
    freq_dict = {u: c / n for u, c in zip(unique, counts)}

    # node features
    node_features = []
    for res_name in res_names:
        props = AMINO_ACID_PROPERTIES.get(res_name, AMINO_ACID_PROPERTIES['UNK'])
        features = torch.tensor([freq_dict.get(res_name, 0.0),
                                 props['hydrophobicity'],
                                 props['mw']], dtype=torch.float)
        node_features.append(features)
    x = torch.stack(node_features)

    # edges
    kdtree = cKDTree(coords)
    pairs = kdtree.query_pairs(r=5.0)
    edge_index, edge_attr = [], []
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([[dist], [dist]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float),
        y=torch.tensor([label]),
        name=structure.id,
        chain_ids=chain_ids
    )

# -------------------------
# Main pipeline
# -------------------------
if __name__ == "__main__":
    with open("batching/node.json", "r") as f:
        uniprot_ids = json.load(f)

    datalist = []
    for uniprot_id in uniprot_ids:
        pdb_path = fetch_pdb(uniprot_id)
        if pdb_path:
            data = pdb_to_data(pdb_path, label=0)
            if data:
                datalist.append(data)

    # save processed list
    with open("protein_graphs.pkl", "wb") as f:
        pickle.dump(datalist, f)

    print(f"📦 Saved {len(datalist)} protein Data objects → protein_graphs.pkl")
    print(datalist)

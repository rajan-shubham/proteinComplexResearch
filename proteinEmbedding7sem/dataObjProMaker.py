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
alphafold_api_url = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
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
    'HOH': {'hydrophobicity': -0.4, 'mw': 18.02}, # Water
    '': {'hydrophobicity': 0, 'mw': 0}, # For unknown residues
      }
    AMINO_ACID_TYPES = list(AMINO_ACID_PROPERTIES.keys())
    NUM_FEATURES = len(AMINO_ACID_TYPES) + 2  # One-hot encoding (21) + 2 properties


    ca_atoms = []
    residues = []
    for residue in structure.get_residues():
        if "CA" in residue:
            ca_atoms.append(residue["CA"])
            residues.append(residue)

    n = len(ca_atoms)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(ca_atoms[i].coord - ca_atoms[j].coord)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Create node features
    node_features = []
    for residue in residues:
        res_name = residue.get_resname()
        
        # One-hot encode the amino acid type
        one_hot = torch.zeros(len(AMINO_ACID_TYPES))
        try:
            idx = AMINO_ACID_TYPES.index(res_name)
            one_hot[idx] = 1
        except ValueError:
            pass # Keep all zeros for unknown residues
            
        # Get numerical properties
        props = AMINO_ACID_PROPERTIES.get(res_name, AMINO_ACID_PROPERTIES[''])
        hydrophobicity = props['hydrophobicity']
        mw = props['mw']
        
        # Combine all features into a single tensor
        features = torch.cat((one_hot, torch.tensor([hydrophobicity, mw], dtype=torch.float)))
        node_features.append(features)

    x = torch.stack(node_features)

    # Create edge index and edge attributes based on distance matrix
    threshold = 5.0
    edge_index = []
    edge_attr = []

    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i, j] < threshold:
                edge_index.append([i, j])
                edge_attr.append([dist_matrix[i, j]])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create the graph object
    coords = np.vstack([atom.coord for atom in ca_atoms])  # shape [n,3]
    chain_ids = [res.get_parent().id for res in residues]  # chain IDs


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float),
        y=torch.tensor([0]),  # placeholder label
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

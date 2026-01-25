from Bio.PDB import *
import numpy as np
import torch
from torch_geometric.data import Data

parser = PDBParser()
structure = parser.get_structure("haemoglobin", "/Users/rajan/github/proteinComplex/sqlProtein/1a3n.pdb")
# structure = parser.get_structure("cellGrowthProtein", "/Users/rajan/github/proteinComplex/sqlProtein/P00533.pdb")

# Amino acid properties
# One-hot encoding for 20 standard amino acids, plus a few properties
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
print(ca_atoms[0].coord)

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

unique_chains = list(set(chain_ids))

print(unique_chains)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    pos=torch.tensor(coords, dtype=torch.float),
    y=torch.tensor([0]),  # placeholder label
    name=structure.id,
    chain_ids=chain_ids
)


print(data)
print("Node features tensor (x):", data.x.shape)
print(data.x)
print("Edge index tensor:", data.edge_index.shape)
print(data.edge_index)
print("Edge attributes tensor:", data.edge_attr.shape)
print(data.edge_attr)
print("Chains example:", chain_ids[300:310])


"""
[-84.082  -7.267   5.658]
Data(x=[1210, 24], edge_index=[2, 2986], edge_attr=[2986, 1])
Node features tensor (x): torch.Size([1210, 24])
tensor([[  0.0000,   0.0000,   0.0000,  ...,   0.0000,   1.9000, 149.2100],
        [  0.0000,   1.0000,   0.0000,  ...,   0.0000,  -4.5000, 174.2000],
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,  -1.6000, 115.1300],
        ...,
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,   4.5000, 131.1800],
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,  -0.4000,  75.0700],
        [  1.0000,   0.0000,   0.0000,  ...,   0.0000,   1.8000,  89.0900]])
Edge index tensor: torch.Size([2, 2986])
tensor([[   0,    1,    1,  ..., 1208, 1208, 1209],
        [   1,    0,    2,  ..., 1207, 1209, 1208]])
Edge attributes tensor: torch.Size([2986, 1])
tensor([[3.8880],
        [3.8880],
        [3.8726],
        ...,
        [3.1084],
        [3.8526],
        [3.8526]])"""
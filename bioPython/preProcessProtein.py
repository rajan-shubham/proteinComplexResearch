from Bio.PDB import PDBParser
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

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
    'HOH': {'hydrophobicity': -0.4, 'mw': 18.02}, # water
    'UNK': {'hydrophobicity': 0.0, 'mw': 0.0}     # unknown residues
}

# -------------------------
# Parse structure
# -------------------------
parser = PDBParser(QUIET=True)
structure = parser.get_structure("haemoglobin", "/Users/rajan/github/proteinComplex/sqlProtein/1a3n.pdb")
# structure = parser.get_structure("cellGrowthProtein", "/Users/rajan/github/proteinComplex/sqlProtein/P00533.pdb")

# Collect CA atoms and residues
ca_atoms = []
residues = []
chain_ids = []

for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                ca_atoms.append(residue["CA"])
                residues.append(residue)
                chain_ids.append(chain.id)

n = len(ca_atoms)
coords = np.vstack([atom.coord for atom in ca_atoms])  # shape [n,3]

# -------------------------
# Compute residue composition (frequency)
# -------------------------
res_names = [res.get_resname() if res.get_resname() in AMINO_ACID_PROPERTIES else 'UNK'
             for res in residues]
unique, counts = np.unique(res_names, return_counts=True)
freq_dict = {u: c / n for u, c in zip(unique, counts)}  # frequency per residue

# -------------------------
# Node features
# -------------------------
node_features = []
for res_name in res_names:
    props = AMINO_ACID_PROPERTIES.get(res_name, AMINO_ACID_PROPERTIES['UNK'])
    hydrophobicity = props['hydrophobicity']
    mw = props['mw']
    freq = freq_dict.get(res_name, 0.0)
    features = torch.tensor([freq, hydrophobicity, mw], dtype=torch.float)
    node_features.append(features)

x = torch.stack(node_features)  # shape [n,3]

# -------------------------
# Build edges using KDTree (optimized)
# -------------------------
threshold = 5.0
kdtree = cKDTree(coords)
pairs = kdtree.query_pairs(r=threshold)  # set of (i,j) with i<j

edge_index = []
edge_attr = []
for i, j in pairs:
    dist = np.linalg.norm(coords[i] - coords[j])
    # add both directions (i->j, j->i)
    edge_index.extend([[i, j], [j, i]])
    edge_attr.extend([[dist], [dist]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

# -------------------------
# Build Data object
# -------------------------
data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    pos=torch.tensor(coords, dtype=torch.float),
    y=torch.tensor([0]),  # placeholder label (change as needed)
    name=structure.id,
    chain_ids=chain_ids
)

# -------------------------
# Debug prints
# -------------------------
print(data)
print("Node features (x):", data.x.shape)
print(data.x[:5])
print("Edge index:", data.edge_index.shape)
print("Edge attr:", data.edge_attr.shape)
print("first ca ca length:", data.edge_attr[0:10])
print("Position tensor (pos):", data.pos.shape)
print("Chains example:", chain_ids[300:310])

"""
print(res_names)
print(unique)
print(counts)
print(freq_dict) 
"""
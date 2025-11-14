# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Original file is located at
    https://colab.research.google.com/drive/16i4ZMbWDEPtuvFq5ZhEQ4rvq6icsoLQ3
"""

# 1. Load the dataset
# 2. Basic preprecessing (80:20) traing test data set
# 3. Training Process (in loof or no. of epochs)
  # a. Create the Model
  # b. Forward pass
  # c. Loss Calculation
  # d. Back propogation
  # e. Parameters update
# 4. Model evaluation
!pip install torch_geometric

import json
import os
import requests
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
# from Bio.PDB import PDBParser
import pickle

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

with open("protein_graphs.pkl", "rb") as f:
    datalist = pickle.load(f)

# finding length of datalist
# Assuming your list is named data_list
print("--- Inspecting Data Objects ---")
for i, data in enumerate(datalist):
    print(data)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. EXAMPLE DATA (your list)
# ----------------------------

# Assume you already have your list of protein graphs
# Example:
# datalist = [Data(...), Data(...), Data(...), ...]
# Each Data: x=[N,24], edge_index=[2,E], y=[1]

# Example synthetic list for testing (replace this with your actual list)
from torch_geometric.data import Data
import torch


print(datalist is not None)
# datalist = [
#     Data(x=torch.randn(400, 24), edge_index=torch.randint(0, 400, (2, 856)), y=torch.tensor([1])),
#     Data(x=torch.randn(502, 24), edge_index=torch.randint(0, 502, (2, 1158)), y=torch.tensor([1])),
#     Data(x=torch.randn(756, 24), edge_index=torch.randint(0, 756, (2, 1680)), y=torch.tensor([0])),
#     Data(x=torch.randn(141, 24), edge_index=torch.randint(0, 141, (2, 376)), y=torch.tensor([0])),
#     Data(x=torch.randn(208, 24), edge_index=torch.randint(0, 208, (2, 504)), y=torch.tensor([1])),
#     Data(x=torch.randn(1613, 24), edge_index=torch.randint(0, 1613, (2, 3954)), y=torch.tensor([0])),
# ]

# ----------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------
train_list, test_list = train_test_split(datalist, test_size=0.3, random_state=42)

print(f"Train size: {len(train_list)}, Test size: {len(test_list)}")

# ----------------------------
# 3. MODEL DEFINITION
# ----------------------------
class ProteinGCN(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=64, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- GCN layers ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # --- Pooling ---
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_channels)

        # --- MLP head ---
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))  # Binary output (0-1)
        return x

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.lin1, self.lin2]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# ----------------------------
# 4. TRAINING SETUP
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()

# ----------------------------
# 5. TRAINING LOOP
# ----------------------------
def train_model(model, data_list, optimizer, criterion, epochs=30, batch_size=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle and batch data in groups of 5 proteins
        torch.manual_seed(epoch)
        perm = torch.randperm(len(data_list))
        data_list = [data_list[i] for i in perm]

        for i in range(0, len(data_list), batch_size):
            batch_graphs = data_list[i:i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            optimizer.zero_grad()
            out = model(batch)  # (num_graphs, 1)

            # Combine their probabilities into a mean for complex prediction
            pred_complex = out.mean()  # scalar between 0-1

            # FIXED: convert labels to float tensor before mean()
            label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs], dtype=torch.float32)
            label_complex = torch.tensor([1.0]) if label_vals.mean() > 0.5 else torch.tensor([0.0])
            label_complex = label_complex.to(device).view(1, 1)

            loss = criterion(pred_complex.view(1, 1), label_complex)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

# ----------------------------
# 6. EVALUATION FUNCTION
# ----------------------------
@torch.no_grad()
def evaluate(model, data_list, batch_size=5):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(data_list), batch_size):
        batch_graphs = data_list[i:i + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)

        out = model(batch)
        pred_complex = out.mean().item()
        pred_label = 1 if pred_complex > 0.5 else 0

        # FIXED: convert labels to float tensor before mean()
        label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs], dtype=torch.float32)
        true_label = 1 if label_vals.mean() > 0.5 else 0

        correct += int(pred_label == true_label)
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# ----------------------------
# 7. RUN TRAINING & EVAL
# ----------------------------
train_model(model, train_list, optimizer, criterion, epochs=30)
# print(model.conv1.out_channels)
# print(model.conv1.bias)
# print(model.lin1.weight)
# print(model.lin1.bias)
evaluate(model, test_list)

!pip install torchinfo

from torchinfo import summary

summary(model, input_size=(192, 24))


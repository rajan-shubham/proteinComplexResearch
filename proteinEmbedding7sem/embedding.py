# -*- coding: utf-8 -*-
"""embedding.ipynb

Original file is located at
    https://colab.research.google.com/drive/10lKf3rEl_9GuMtf58X9fMKaR5NPvki7F
"""

# !pip install torch_geometric

import os

os.listdir('.')

import pickle

with open('protein_graphs.pkl', 'rb') as f:
    protein_graphs = pickle.load(f)

for data in protein_graphs:
  print(data)
print(len(protein_graphs))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool
from torch_geometric.loader import DataLoader

loader = DataLoader(
    protein_graphs,
    batch_size=4,
    shuffle=False
)

class GAT1(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=512, heads=1):
        super(GAT1, self).__init__()

        # Initial projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # GAT layers (edge-aware)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)

        # Pooling layers
        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        # Initial projection
        x = self.fc1(x)

        # ---- Block 1 ----
        x_res = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x + x_res)

        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )

        # ---- Block 2 ----
        x_res = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )

        # ---- Block 3 ----
        x_res = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res)

        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )

        # ---- Protein-level embedding ----
        protein_embedding = global_mean_pool(x, batch)

        return protein_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAT1(
    input_dim=24,
    hidden_dim=512
).to(device)

model.eval()

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        emb = model(batch)

        print("Protein embedding shape:", emb.shape)
        # shape = [batch_size, 2432]
        break

print(emb[0])


# -*- coding: utf-8 -*-
"""ProteinInteractionModel.ipynb
Original file is located at
    https://colab.research.google.com/drive/1szCbx06ompxQjBLBdCVlg3cNgv9BeZr-
"""

# Protein Graph (x, edge_index, edge_attr)
#         ↓
# Edge-aware GNN (NNConv)
#         ↓
# Protein embedding (fixed vector)
#         ↓
# Pairwise fusion (GCN-style MLP)
#         ↓
# Binary PPI prediction (HIPPIE)
# !pip install torch_geometric

import os

os.listdir('.')

import pickle

with open('protein_graphs.pkl', 'rb') as f:
    protein_graphs = pickle.load(f)

protein_dict = {}

for data in protein_graphs:
    # extract UniProt ID
    uniprot_id = data.name.replace('.pdb', '')
    protein_dict[uniprot_id] = data

print("Proteins mapped:", len(protein_dict))
print("Sample keys:", list(protein_dict.keys())[:5])

for g in protein_graphs:
    g.edge_attr = g.edge_attr / 10.0

import pandas as pd

hippie_df = pd.read_csv(
    'hippie_protein_interactions.csv',
    header=None
)

hippie_df = hippie_df[hippie_df[0] != '0']
hippie_df = hippie_df[hippie_df[1] != '1']

hippie_df.reset_index(drop=True, inplace=True)
hippie_df.head()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool

class ProteinEncoder(nn.Module):
    def __init__(self, node_dim=24, hidden_dim=128, emb_dim=256):
        super().__init__()

        # ---------- FIX 1: SEPARATE EDGE NETWORKS ----------

        # For conv1: node_dim → hidden_dim
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, node_dim * hidden_dim)
        )

        # For conv2: hidden_dim → hidden_dim
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim)
        )

        self.conv1 = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp1,
            aggr='mean'
        )

        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp2,
            aggr='mean'
        )

        self.lin_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # ---------- FIX 2: NORMALIZE EDGE ATTR ----------
        # edge_attr shape: [num_edges, 1]
        # edge_attr = edge_attr / 10.0

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Graph-level embedding
        x = global_mean_pool(x, batch)
        x = self.lin_proj(x)

        return x  # [batch_size, emb_dim]

class PPIClassifier(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        pair_dim = emb_dim * 4  # concat, diff, product

        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, hA, hB):
        pair_feat = torch.cat([
            hA,
            hB,
            torch.abs(hA - hB),
            hA * hB
        ], dim=1)

        logits = self.classifier(pair_feat)
        return logits.squeeze(1)

class ProteinInteractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ProteinEncoder()
        self.ppi_head = PPIClassifier()

    def forward(self, dataA, dataB):
        hA = self.encoder(dataA)
        hB = self.encoder(dataB)
        return self.ppi_head(hA, hB)

available_proteins = set(protein_dict.keys())

hippie_filtered = hippie_df[
    hippie_df[0].isin(available_proteins) &
    hippie_df[1].isin(available_proteins)
]

print("Filtered interactions:", len(hippie_filtered))

positive_pairs = list(
    zip(hippie_filtered[0].values, hippie_filtered[1].values)
)

print("Positive pairs:", len(positive_pairs))

import random

all_proteins = list(available_proteins)
positive_set = set(positive_pairs)

def sample_negative_pairs(n):
    neg = set()
    while len(neg) < n:
        a, b = random.sample(all_proteins, 2)
        if (a, b) not in positive_set and (b, a) not in positive_set:
            neg.add((a, b))
    return list(neg)

negative_pairs = sample_negative_pairs(len(positive_pairs))
print("Negative pairs:", len(negative_pairs))

from sklearn.model_selection import train_test_split

pairs = positive_pairs + negative_pairs
labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

train_pairs, test_pairs, y_train, y_test = train_test_split(
    pairs, labels, test_size=0.2, random_state=42, stratify=labels
)

train_pairs, val_pairs, y_train, y_val = train_test_split(
    train_pairs, y_train, test_size=0.1, random_state=42, stratify=y_train
)

from torch.utils.data import Dataset
from torch_geometric.data import Batch

class PPIDataset(Dataset):
    def __init__(self, pairs, labels, protein_dict):
        self.pairs = pairs
        self.labels = labels
        self.graphs = protein_dict

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return self.graphs[a], self.graphs[b], y



def ppi_collate(batch):
    dataA, dataB, y = zip(*batch)
    return (
        Batch.from_data_list(dataA),
        Batch.from_data_list(dataB),
        torch.stack(y)
    )

# Datasets
train_dataset = PPIDataset(train_pairs, y_train, protein_dict)
val_dataset   = PPIDataset(val_pairs, y_val, protein_dict)
test_dataset  = PPIDataset(test_pairs, y_test, protein_dict)

# DataLoaders
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=ppi_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=ppi_collate
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=ppi_collate
)

A, B, y = next(iter(train_loader))
print(A)
print(B)
print(y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProteinInteractionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

def train_epoch(loader):
    model.train()
    total_loss = 0

    for dataA, dataB, y in loader:
        dataA = dataA.to(device)
        dataB = dataB.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(dataA, dataB)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

from sklearn.metrics import roc_auc_score

def evaluate(loader):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for dataA, dataB, y in loader:
            logits = model(
                dataA.to(device),
                dataB.to(device)
            )
            probs = torch.sigmoid(logits)

            preds.append(probs.cpu())
            ys.append(y)

    return roc_auc_score(
        torch.cat(ys).numpy(),
        torch.cat(preds).numpy()
    )

EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(train_loader)
    val_auc = evaluate(val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

test_auc = evaluate(test_loader)
print("Test ROC-AUC:", test_auc)

# -*- coding: utf-8 -*-
"""edgeAttrGCN.ipynb

Original file is located at
    https://colab.research.google.com/drive/14kFPtIWB69yeC4-lYxBtKjRnoAHnt_gK
"""

# !pip install torch_geometric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import pickle

with open("protein_graphs.pkl", "rb") as f:
    datalist = pickle.load(f)
# ----------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------
train_list, test_list = train_test_split(datalist, test_size=0.3, random_state=42)

print(f"Train size: {len(train_list)}, Test size: {len(test_list)}")

# ============================================================
# 1. Custom Weighted GCNConv (supports edge_attr)
# ============================================================
class WeightedGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        device = x.device

        # ----------------------------
        # Build weighted adjacency: A[i,j] = weight
        # ----------------------------
        row, col = edge_index
        A = torch.zeros((N, N), device=device)
        A[row, col] = edge_attr.squeeze()

        # Add self-loops with weight = 1.0
        A = A + torch.eye(N, device=device)

        # ----------------------------
        # Normalize A:  D^{-1/2} A D^{-1/2}
        # ----------------------------
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        # ----------------------------
        # GCN propagation:  X' = A_norm X W
        # ----------------------------
        x = self.lin(x)
        return A_norm @ x


# ============================================================
# 2. ProteinGCN using Weighted GCN layers
# ============================================================
class ProteinGCN(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=64, out_channels=1):
        super().__init__()

        self.conv1 = WeightedGCNConv(in_channels, hidden_channels)
        self.conv2 = WeightedGCNConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, out_channels)

    def forward(self, data):
        x, edge_index, batch, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )

        # ---- Weighted GCN layers ----
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # ---- Global pooling ----
        x = global_mean_pool(x, batch)

        # ---- MLP output ----
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

# ============================================================
# 4. TRAINING SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()

# ============================================================
# 5. TRAINING LOOP
# ============================================================
def train_model(model, data_list, optimizer, criterion, epochs=30, batch_size=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        torch.manual_seed(epoch)
        perm = torch.randperm(len(data_list))
        data_list = [data_list[i] for i in perm]

        for i in range(0, len(data_list), batch_size):
            batch_graphs = data_list[i:i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            optimizer.zero_grad()

            out = model(batch)   # (num_graphs,1)
            pred_complex = out.mean()

            label_vals = torch.tensor(
                [float(d.y.item()) for d in batch_graphs],
                dtype=torch.float32,
            )
            label_complex = torch.tensor([1.0]) if label_vals.mean() > 0.5 else torch.tensor([0.0])
            label_complex = label_complex.to(device).view(1, 1)

            loss = criterion(pred_complex.view(1, 1), label_complex)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")


# ============================================================
# 6. EVALUATE
# ============================================================
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

        label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs])
        true_label = 1 if label_vals.mean() > 0.5 else 0

        correct += int(pred_label == true_label)
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# ============================================================
# 7. RUN TRAINING + TEST
# ============================================================
train_model(model, train_list, optimizer, criterion, epochs=30)
evaluate(model, test_list)


# -*- coding: utf-8 -*-
"""ProteinGraphSAGE.ipynb

Original file is located at
    https://colab.research.google.com/drive/1N89mi-4VSMR-8-RKPAOiDYVZNWv0cEbK
"""

# !pip install torch_geometric

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load data
# ----------------------------
with open("protein_graphs.pkl", "rb") as f:
    datalist = pickle.load(f)

train_list, test_list = train_test_split(datalist, test_size=0.3, random_state=42)
print(f"Train size: {len(train_list)}, Test size: {len(test_list)}")

# ----------------------------
# 2. Model (GraphSAGE)
# ----------------------------
class ProteinGraphSAGE(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=64, out_channel=1):
        super().__init__()
        # Two-layer GraphSAGE
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)

        # MLP head
        self.lin1 = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, out_channel)
        self.dropout = 0.2

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Note: edge_attr is ignored here. To use edge attributes, you can expand node features
        # or implement custom aggregation that uses edge features.
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.sage2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # graph embedding

        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

    def reset_parameters(self):
        for layer in [self.sage1, self.sage2, self.lin1, self.lin2]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

# ----------------------------
# 3. Training / Eval utilities
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinGraphSAGE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()

def train_model(model, data_list, optimizer, criterion, epochs=30, batch_size=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        torch.manual_seed(epoch)
        perm = torch.randperm(len(data_list))
        data_list = [data_list[i] for i in perm]

        for i in range(0, len(data_list), batch_size):
            batch_graphs = data_list[i:i+batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)

            optimizer.zero_grad()
            out = model(batch)  # (num_graphs, 1)
            pred_complex = out.mean()

            label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs], dtype=torch.float32)
            label_complex = torch.tensor([1.0]) if label_vals.mean() > 0.5 else torch.tensor([0.0])
            label_complex = label_complex.to(device).view(1,1)

            loss = criterion(pred_complex.view(1,1), label_complex)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.6f} ---> last_batch_loss={loss.item():.6e}")

@torch.no_grad()
def evaluate(model, data_list, batch_size=5):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(data_list), batch_size):
        batch_graphs = data_list[i:i+batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)

        out = model(batch)
        pred_complex = out.mean().item()
        pred_label = 1 if pred_complex > 0.5 else 0

        label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs], dtype=torch.float32)
        true_label = 1 if label_vals.mean() > 0.5 else 0

        correct += int(pred_label == true_label)
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# ----------------------------
# 4. Run
# ----------------------------
if __name__ == "__main__":
    print("Starting GraphSAGE training...")
    model.reset_parameters()
    train_model(model, train_list, optimizer, criterion, epochs=30, batch_size=5)
    print("Evaluating on test set...")
    evaluate(model, test_list)


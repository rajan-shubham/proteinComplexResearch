# -*- coding: utf-8 -*-
"""ProteinGCN2.ipynb

Original file is located at
    https://colab.research.google.com/drive/17GLgIQIxBc84h3J8pgQjejn8ELsSSHQq
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

class ProteinGCN2(nn.Module):
  def __init__(self, in_channels=24, hidden_channels=64, out_channel=1):
    super().__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.lin1 = nn.Linear(hidden_channels, 32)
    self.lin2 = nn.Linear(32, out_channel)

  def forward(self, data):
    x, edge_index, batch, edge_attr = (
          data.x,
          data.edge_index,
          data.batch,
          data.edge_attr,
      )
    # If using edge_attr (e.g. distances) as weights:
    if edge_attr is not None:
      # edge_attr might be shape [E,1], we squeeze to [E]
      edge_attr = edge_attr.view(-1)

    # --- GCN layers ---
    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = self.conv2(x, edge_index, edge_attr)
    x = F.relu(x)

    # Global pooling: average node features to get graph representaton
    x = global_mean_pool(x, batch)

    # --- ANN/MLP head ---
    x = F.relu(self.lin1(x))
    x = torch.sigmoid(self.lin2(x))
    return x

  def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.lin1, self.lin2]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# ----------------------------
# 4. TRAINING SETUP
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinGCN2().to(device)
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

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} ---> {loss.item()}")

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

# ============================================================
# 7. RUN TRAINING + TEST
# ============================================================
train_model(model, train_list, optimizer, criterion, epochs=30)
evaluate(model, test_list)




"""
Here is a clean, honest, and **expert-level review** of your code for **protein complex prediction using GCN + graph pooling + ANN**, including **what is correct**, **what is incorrect**, and **what must be improved** for the model to be *scientifically meaningful*.

---

# ✅ **WHAT IS CORRECT IN YOUR CODE**

### ✔️ 1. **GCN Model Architecture**

Your `ProteinGCN2` class is correct for:

* Node feature input (`x`)
* Two GCN layers
* Global mean pooling to compute graph-level embedding
* Small MLP head for final probability

This is a valid graph classification model.

---

### ✔️ 2. **Batching using `Batch.from_data_list`**

You correctly batch multiple protein graphs for training.

---

### ✔️ 3. **Handling edge attributes**

You correctly treat `edge_attr` (distances, bonds, etc.) as weights and reshape them.

---

### ✔️ 4. **Forward pass logic**

Everything in the forward pass is clean and correct.

---

# ⚠️ **WHAT IS NOT SCIENTIFICALLY CORRECT**

Here is the **major issue**:

## ❌ You are averaging *the predictions of individual proteins* and treating it as the probability of "complex formation".

### Why this is wrong

A protein complex is made of **multiple interacting proteins**.

But your model sees each protein **independently**, not jointly.

You take:

```
out = model(batch)  # → predictions for individual proteins
pred_complex = out.mean()
```

This means:

**"A complex exists IF the proteins individually have high 'contribution probability'"**

This is NOT a correct biological assumption.

A complex cannot be predicted based on average individual protein propensity.

You are missing the **inter-protein interaction graph**.

---

# ❌ **MAIN SCIENTIFIC PROBLEM**

### Your datalist = list of individual protein graphs

But *protein complex prediction requires modelling interactions BETWEEN proteins*, not only individual structures.

This model cannot learn complex-level patterns because:

* No edges between proteins
* No combined graph
* No co-evolution/node matching
* No interface residue prediction
* No docking geometry
* No contact probability map

---

# ⚠️ LABEL PROBLEM (VERY IMPORTANT)

Your labels:

```python
label_vals = torch.tensor([float(d.y.item()) for d in batch_graphs])
true_label = 1 if label_vals.mean() > 0.5 else 0
```

This means:

* Each **protein** has a label
* You convert per-protein labels → per-complex label via **mean > 0.5**

This is **mathematically wrong** and **biologically meaningless**.

A true complex label must be **given per complex**, NOT per individual protein.

---

# ⚠️ TRAINING LOSS PROBLEM

You compute loss as:

```
pred_complex = out.mean()
label_complex = 1 or 0
loss = BCELoss(pred_complex, label_complex)
```

Meaning:

* Every batch of proteins results in **1 scalar loss**
* Gradients update GCN as if proteins individually should move to match average

This destroys signal and prevents learning.

---

# ⚠️ EVALUATION IS ALSO INVALID

Evaluation uses the same averaging trick → so accuracy results are meaningless.

---

# 🚀 **HOW TO MAKE IT SCIENTIFICALLY CORRECT**

If you want **real protein complex prediction**, you need one of these solutions:

---

## ✅ **Solution A — Build a Graph-of-Proteins Model**

Combine individual protein graphs into a *supra-graph*:

* Nodes = residues from all proteins
* Add inter-protein edges based on:

  * Spatial proximity (from docking)
  * AlphaFold-Multimer predicted contact map
  * Sequence co-evolution
  * Known interface positions

Then train a single GNN on the combined complex graph.

---

## ✅ **Solution B — Siamese / Matching GNN**

If complex = (Protein A, Protein B):

* Encode each protein separately using GCN → embedding A, embedding B
* Combine using:

  * cosine similarity
  * bilinear layer
  * cross-attention
* Predict probability of forming a complex

---

## ✅ **Solution C — Residue-level Cross Attention**

Use transformer-like cross attention between two proteins.

---

## 🚀 **If you want, I can write the correct version for:**

✔️ Pairwise protein complex prediction
✔️ Multi-protein complex prediction
✔️ Graph-of-graphs complex prediction
✔️ AlphaFold-Multimer style interface prediction
✔️ Co-evolution feature pipelines

---

# ⭐ Final Verdict

### ❌ **Your code is syntactically correct, but biologically incorrect.**

It cannot perform real protein complex prediction because:

* It treats proteins independently
* It averages independent predictions
* Labels are incorrectly aggregated
* No inter-protein interactions
* Wrong training signal

### ✔️ The model **can** learn “protein-level contribution scores”, **not** complex prediction.

---

# Want the corrected version?

I can generate any of the following complete working implementations:

### 🔥 Option 1: **Siamese GCN for 2-protein complex prediction**

### 🔥 Option 2: **Multi-Protein Supra-Graph GNN**

### 🔥 Option 3: **Graph-of-Graphs Protein Complex classifier**

### 🔥 Option 4: **AlphaFold-style interface attention GNN**
"""
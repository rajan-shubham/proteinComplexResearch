# -*- coding: utf-8 -*-
"""
Full PPI + Protein Complex Prediction Pipeline
=============================================
Upload these files to Colab before running:
  - proteinGraphsIndexed.pkl
  - protein_index_map.json
  - positiveEdges_indexed.csv       (your positive_edge_1024.csv, already indexed)
  - combined_stringent.txt          (Negatome: https://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_stringent.txt)
  - DT1_Dict_hippie.json
"""

# ================= INSTALL =================
# Run this cell first in Colab:
# !pip install torch-geometric

# ================= IMPORTS =================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import random
import numpy as np

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, SAGPooling
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# =====================================================
# CONFIG
# =====================================================

SUBSET = 1500          # use 1/5 of data for Colab
HIDDEN_DIM = 128       # reduced from 512 to fit Colab RAM
EMBED_DIM = 64
HEADS = 1
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 256       # edges per batch
NEG_RATIO = 1          # negative : positive ratio
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================
# STEP 1 — LOAD PROTEIN GRAPHS
# =====================================================

with open("/content/proteinGraphsIndexed.pkl", "rb") as f:
    protein_graphs = pickle.load(f)

protein_graphs = protein_graphs[:SUBSET]
print(f"Loaded {len(protein_graphs)} protein graphs")

# =====================================================
# STEP 2 — BUILD PROTEIN ID → INDEX MAP
# =====================================================
# Your graphs already have sequential integer indices
# but we also need UniProt → index for Negatome matching.

with open("/content/protein_index_map.json") as f:
    protein_index_map = json.load(f)   # { "Q92769": 0, "P26373": 1, ... }

# Restrict map to proteins in the subset
valid_indices = set(range(SUBSET))
uniprot2idx = {k: v for k, v in protein_index_map.items() if v in valid_indices}

print(f"Proteins in index map (subset): {len(uniprot2idx)}")

# =====================================================
# STEP 3 — LOAD POSITIVE PPI EDGES (already indexed)
# =====================================================

positive_edges = []
with open("/content/positiveEdges_indexed.csv") as f:
    next(f)
    for line in f:
        a, b = line.strip().split(",")
        a, b = int(a), int(b)
        if a < SUBSET and b < SUBSET:
            positive_edges.append((a, b))

print(f"Positive edges (subset): {len(positive_edges)}")

# =====================================================
# STEP 4 — LOAD NEGATIVE EDGES FROM NEGATOME
# =====================================================
# combined_stringent.txt uses UniProt IDs separated by tab.

negatome_edges = []
try:
    with open("/content/combined_stringent.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            a_uniprot, b_uniprot = parts[0], parts[1]
            # Strip isoform suffixes like -3
            a_uniprot = a_uniprot.split("-")[0]
            b_uniprot = b_uniprot.split("-")[0]
            if a_uniprot in uniprot2idx and b_uniprot in uniprot2idx:
                i, j = uniprot2idx[a_uniprot], uniprot2idx[b_uniprot]
                if i != j:
                    negatome_edges.append((i, j))
    print(f"Negatome negative edges matched to subset: {len(negatome_edges)}")
except FileNotFoundError:
    print("WARNING: combined_stringent.txt not found — using only random negatives.")

# =====================================================
# STEP 5 — RANDOM NEGATIVE SAMPLING (top-up)
# =====================================================
# Generate random negatives to reach NEG_RATIO * len(positive_edges)

positive_set = set(positive_edges) | {(b, a) for a, b in positive_edges}
target_neg = NEG_RATIO * len(positive_edges)
random_negs = set((i, j) for i, j in negatome_edges)

max_attempts = target_neg * 20
attempts = 0
while len(random_negs) < target_neg and attempts < max_attempts:
    i = random.randint(0, SUBSET - 1)
    j = random.randint(0, SUBSET - 1)
    if i != j and (i, j) not in positive_set and (i, j) not in random_negs:
        random_negs.add((i, j))
    attempts += 1

negative_edges = list(random_negs)[:target_neg]
print(f"Total negative edges used: {len(negative_edges)}")

# =====================================================
# STEP 6 — BUILD FULL PPI EDGE INDEX (for GCN)
# =====================================================
# GCN needs the full positive PPI graph as its topology.

all_ppi = []
for a, b in positive_edges:
    all_ppi.append([a, b])
    all_ppi.append([b, a])

ppi_edge_index = torch.tensor(all_ppi, dtype=torch.long).t().contiguous().to(device)
print(f"PPI edge_index shape: {ppi_edge_index.shape}")

# =====================================================
# STEP 7 — TRAIN / VAL / TEST SPLITS
# =====================================================

def split_edges(edges, train=0.70, val=0.15):
    random.shuffle(edges)
    n = len(edges)
    t = int(n * train)
    v = int(n * (train + val))
    return edges[:t], edges[t:v], edges[v:]

pos_train, pos_val, pos_test = split_edges(positive_edges)
neg_train, neg_val, neg_test = split_edges(negative_edges)

print(f"Train  — pos: {len(pos_train)}, neg: {len(neg_train)}")
print(f"Val    — pos: {len(pos_val)},   neg: {len(neg_val)}")
print(f"Test   — pos: {len(pos_test)},  neg: {len(neg_test)}")

def make_batches(pos, neg, batch_size):
    """Yields (edge_pairs, labels) batches."""
    pairs  = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
    random.shuffle(pairs)
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        src   = torch.tensor([p[0] for p in chunk], dtype=torch.long)
        dst   = torch.tensor([p[1] for p in chunk], dtype=torch.long)
        lbl   = torch.tensor([p[2] for p in chunk], dtype=torch.float)
        yield src, dst, lbl

# =====================================================
# MODELS  (minimal changes from your original)
# =====================================================

class GAT1(nn.Module):
    """Structure-aware encoder for a single protein graph."""

    def __init__(self, input_dim=24, hidden_dim=HIDDEN_DIM, heads=HEADS):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1)
        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr

        # dummy batch: all nodes belong to one graph
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.fc1(x)

        # Layer 1
        x_res = x
        x     = self.conv1(x, edge_index, edge_attr=edge_attr)
        x     = self.bn1(x)
        x     = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        # Layer 2
        x_res = x
        x     = self.conv2(x, edge_index, edge_attr=edge_attr)
        x     = self.bn2(x)
        x     = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        # Layer 3
        x_res = x
        x     = self.conv3(x, edge_index, edge_attr=edge_attr)
        x     = self.bn3(x)
        x     = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        return global_mean_pool(x, batch)   # [1, hidden_dim]


class JointGAT_GCN(nn.Module):
    """
    Full model:
      GAT1 per protein  →  project 512→64  →  GCN over PPI graph
    The only change from your original: hidden_dim is now a parameter
    so we can reduce it to 128 for Colab.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.gat     = GAT1(hidden_dim=hidden_dim)
        self.project = nn.Linear(hidden_dim, embed_dim)
        self.gcn1    = GCNConv(embed_dim, embed_dim)
        self.gcn2    = GCNConv(embed_dim, embed_dim)

    def encode_all(self, protein_graphs, ppi_edge_index):
        """
        Encode all proteins once → matrix X [N, embed_dim]
        Then refine with GCN over PPI topology.
        """
        embeddings = []
        for g in protein_graphs:
            g   = g.to(device)
            emb = self.gat(g)           # [1, hidden_dim]
            emb = self.project(emb)     # [1, embed_dim]
            embeddings.append(emb)

        X = torch.cat(embeddings, dim=0)             # [N, embed_dim]
        X = F.relu(self.gcn1(X, ppi_edge_index))
        X = self.gcn2(X, ppi_edge_index)            # [N, embed_dim]
        return X

    def forward(self, protein_graphs, ppi_edge_index):
        return self.encode_all(protein_graphs, ppi_edge_index)


# =====================================================
# STEP 8 — TRAINING
# =====================================================

model     = JointGAT_GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def compute_loss_and_scores(z, src, dst, lbl):
    """Dot-product link prediction."""
    scores = (z[src] * z[dst]).sum(dim=1)
    probs  = torch.sigmoid(scores)
    loss   = F.binary_cross_entropy(probs, lbl.to(device))
    return loss, probs.detach().cpu(), lbl.cpu()

def evaluate(z, pos_edges, neg_edges):
    all_probs, all_labels = [], []
    pairs = [(a, b, 1) for a, b in pos_edges] + [(a, b, 0) for a, b in neg_edges]
    if not pairs:
        return 0.0, 0.0
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    lbl = [p[2] for p in pairs]
    with torch.no_grad():
        scores = (z[src] * z[dst]).sum(dim=1)
        probs  = torch.sigmoid(scores).cpu().numpy()
    auc = roc_auc_score(lbl, probs)
    ap  = average_precision_score(lbl, probs)
    return auc, ap

print("\n========== TRAINING ==========")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    steps      = 0

    # --- encode all proteins once per epoch ---
    z = model(protein_graphs, ppi_edge_index)   # [N, 64]

    for src, dst, lbl in make_batches(pos_train, neg_train, BATCH_SIZE):
        optimizer.zero_grad()
        loss, _, _ = compute_loss_and_scores(z, src, dst, lbl)
        loss.backward(retain_graph=True)        # z shares graph; retain for multi-batch
        optimizer.step()
        total_loss += loss.item()
        steps      += 1

    # --- validation (re-encode with no_grad) ---
    model.eval()
    with torch.no_grad():
        z_val = model(protein_graphs, ppi_edge_index)
    val_auc, val_ap = evaluate(z_val, pos_val, neg_val)

    print(f"Epoch {epoch:02d} | Loss: {total_loss/steps:.4f} | "
          f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

# =====================================================
# STEP 9 — TEST EVALUATION
# =====================================================

model.eval()
with torch.no_grad():
    z_final = model(protein_graphs, ppi_edge_index)

test_auc, test_ap = evaluate(z_final, pos_test, neg_test)
print(f"\n===== TEST RESULTS =====")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test AP : {test_ap:.4f}")

# =====================================================
# STEP 10 — SAVE MODEL + EMBEDDINGS
# =====================================================

torch.save(model.state_dict(), "/content/joint_model.pt")
torch.save(z_final, "/content/protein_embeddings.pt")
print("\nSaved model → /content/joint_model.pt")
print("Saved embeddings → /content/protein_embeddings.pt  (shape:", z_final.shape, ")")

# =====================================================
# STEP 11 — PROTEIN COMPLEX PREDICTION
# =====================================================
# DT1_Dict_hippie.json format assumed:
#   { "complex_id": ["UniProtA", "UniProtB", "UniProtC", ...], ... }
#   or a list of lists.

print("\n========== COMPLEX PREDICTION ==========")

with open("/content/DT1_Dict_hippie.json") as f:
    complex_data = json.load(f)

# Support both dict-of-lists and list-of-lists formats
if isinstance(complex_data, dict):
    complexes = list(complex_data.values())
elif isinstance(complex_data, list):
    complexes = complex_data
else:
    raise ValueError("Unexpected DT1_Dict_hippie.json format")

z_cpu = z_final.cpu()

def score_complex(members, uniprot2idx, z, threshold=0.5):
    """
    Compute complex score as sum of pairwise dot-products.
    Returns (score, probability, predicted_label).
    Skips proteins not in the embedding index.
    """
    indices = [uniprot2idx[m] for m in members if m in uniprot2idx]
    if len(indices) < 2:
        return None, None, None

    total = 0.0
    pairs = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total += (z[indices[i]] * z[indices[j]]).sum().item()
            pairs += 1

    score = total / pairs           # average pairwise similarity
    prob  = torch.sigmoid(torch.tensor(score)).item()
    label = 1 if prob > threshold else 0
    return score, prob, label

results = []
skipped = 0
for members in complexes:
    score, prob, label = score_complex(members, uniprot2idx, z_cpu)
    if score is None:
        skipped += 1
        continue
    results.append({
        "members"    : members,
        "score"      : round(score, 4),
        "probability": round(prob,  4),
        "predicted"  : label
    })

print(f"Complexes evaluated : {len(results)}")
print(f"Complexes skipped (not in subset): {skipped}")
print(f"Predicted as complex     : {sum(r['predicted'] for r in results)}")
print(f"Predicted as non-complex : {sum(1 - r['predicted'] for r in results)}")

# Show a sample
print("\n--- Sample predictions (first 10) ---")
for r in results[:10]:
    print(f"  Members: {r['members']} | Score: {r['score']:.3f} | "
          f"Prob: {r['probability']:.3f} | Label: {r['predicted']}")

# Save predictions
with open("/content/complex_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved predictions → /content/complex_predictions.json")

# =====================================================
# SUMMARY
# =====================================================
print("""
===========================================
PIPELINE COMPLETE
===========================================
Files saved:
  /content/joint_model.pt           ← trained model weights
  /content/protein_embeddings.pt    ← z [N, 64] embeddings
  /content/complex_predictions.json ← complex classification results

To reload the model later:
    model = JointGAT_GCN().to(device)
    model.load_state_dict(torch.load('/content/joint_model.pt'))

To reload embeddings:
    z = torch.load('/content/protein_embeddings.pt')
===========================================
""")

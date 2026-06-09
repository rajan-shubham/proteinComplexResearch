# ============================================================
#  PROTEIN COMPLEX PREDICTION — End-to-End Pipeline
#  Copy-paste into Google Colab and run cell by cell
# ============================================================

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install torch_geometric
# !pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
#              -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# !pip install pandas scikit-learn


# ============================================================
# CELL 2 — Imports & Config
# ============================================================
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn import SAGPooling
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score
)

# ---------- hyper-params (tune here) ----------
HIDDEN_DIM  = 128   # GAT hidden dim  (reduce to 64 if OOM)
HEADS       = 4     # GAT attention heads (GATConv uses concat=False → output stays HIDDEN_DIM)
EMBED_DIM   = 64    # projected / GCN embedding dim
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 60
BATCH_SIZE  = 64    # for complex DataLoader
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# CELL 3 — Load raw data
# ============================================================

# ---------- 3a. Protein index map ----------
with open("DataCleaning/protein_index_map.json") as f:
    protein_index_map = json.load(f)          # {uniprot_id: int_idx}
idx_to_pid = {v: k for k, v in protein_index_map.items()}
N_PROTEINS  = len(protein_index_map)
print(f"Proteins in index: {N_PROTEINS}")

# ---------- 3b. Protein graphs (list of PyG Data, ordered by index) ----------
with open("DataCleaning/proteinGraphsIndexed.pkl", "rb") as f:
    protein_graphs = pickle.load(f)           # list[Data], len == N_PROTEINS
# protein_graphs[i] is the graph for protein with integer index i
print(f"Protein graphs loaded: {len(protein_graphs)}")
print("  Example:", protein_graphs[0])

# ---------- 3c. PPI edges (already integer-indexed) ----------
ppi_df   = pd.read_csv("DataCleaning/positiveEdges_indexed.csv")
ppi_src  = ppi_df["Node1"].tolist()
ppi_dst  = ppi_df["Node2"].tolist()
# Build bidirectional edge_index
ppi_src_bi = ppi_src + ppi_dst
ppi_dst_bi = ppi_dst + ppi_src
ppi_edge_index = torch.tensor([ppi_src_bi, ppi_dst_bi], dtype=torch.long).to(device)
print(f"PPI edges (bidirectional): {ppi_edge_index.shape[1]}")

# ---------- 3d. Positive complexes (already integer-indexed) ----------
with open("DataCleaning/indexed_complexes.json") as f:
    pos_complexes_raw = json.load(f)          # list of lists of int indices

# Keep only complexes where ALL members exist in protein_graphs
valid_indices = set(range(len(protein_graphs)))
pos_complexes = [
    cx for cx in pos_complexes_raw
    if len(cx) >= 2 and all(i in valid_indices for i in cx)
]
print(f"Positive complexes (valid): {len(pos_complexes)}")

# ---------- 3e. Negative complexes (NOT yet indexed) ----------
with open("DataCleaning/N_RANDOM_Comb.json") as f:
    neg_complexes_raw = json.load(f)          # list of lists of gene names or indices

# Map negatives: each element must be mappable via protein_index_map
# Handles two formats: already int OR gene-name strings
def map_complex(cx, pid_map, valid_idx):
    """Return list of int indices, or None if any member missing."""
    mapped = []
    for m in cx:
        if isinstance(m, int):
            idx = m
        else:
            idx = pid_map.get(str(m))
            if idx is None:
                return None
        if idx not in valid_idx:
            return None
        mapped.append(idx)
    return mapped if len(mapped) >= 2 else None

neg_complexes = []
for cx in neg_complexes_raw:
    mapped = map_complex(cx, protein_index_map, valid_indices)
    if mapped is not None:
        neg_complexes.append(mapped)

print(f"Negative complexes (after filtering): {len(neg_complexes)}")

# Replicate negatives to match positive count if needed
target = len(pos_complexes)
if len(neg_complexes) < target:
    deficit = target - len(neg_complexes)
    neg_complexes = neg_complexes + random.choices(neg_complexes, k=deficit)
    print(f"  → Replicated to {len(neg_complexes)} negatives")
else:
    neg_complexes = random.sample(neg_complexes, target)   # subsample if surplus
    print(f"  → Subsampled to {len(neg_complexes)} negatives")


# ============================================================
# CELL 4 — Train / Val / Test split (on complexes only)
# ============================================================
# Split: 70% train, 15% val, 15% test
def split_complexes(complexes, label, train_r=0.70, val_r=0.15, seed=42):
    rng = random.Random(seed)
    data = [(cx, label) for cx in complexes]
    rng.shuffle(data)
    n = len(data)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return data[:t1], data[t1:t2], data[t2:]

pos_train, pos_val, pos_test = split_complexes(pos_complexes, 1)
neg_train, neg_val, neg_test = split_complexes(neg_complexes, 0)

train_data = pos_train + neg_train
val_data   = pos_val   + neg_val
test_data  = pos_test  + neg_test

random.shuffle(train_data); random.shuffle(val_data); random.shuffle(test_data)

print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")


# ============================================================
# CELL 5 — Dataset & collate
# ============================================================
class ComplexDataset(Dataset):
    """Each item: (list_of_node_indices, label)."""
    def __init__(self, data): self.data = data
    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]   # (cx_list, label)

def collate_complexes(batch):
    """Returns list of (complex_indices, label) tuples — no padding needed."""
    return batch   # handled manually in training loop

train_loader = TorchDataLoader(
    ComplexDataset(train_data), batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_complexes)
val_loader   = TorchDataLoader(
    ComplexDataset(val_data),   batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_complexes)
test_loader  = TorchDataLoader(
    ComplexDataset(test_data),  batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_complexes)


# ============================================================
# CELL 6 — Models
# ============================================================

# ---- 6a. GAT1: Structure encoder (per protein graph) ----
class GAT1(nn.Module):
    """Structure-aware encoder for a single protein graph."""

    def __init__(self, input_dim=24, hidden_dim=HIDDEN_DIM, heads=HEADS):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        # GATConv with concat=False → output dim == hidden_dim (not heads*hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)
        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x          = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float() if data.edge_attr is not None else None

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.fc1(x)

        # --- Layer 1 ---
        x_res = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x); x = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        # --- Layer 2 ---
        x_res = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x); x = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        # --- Layer 3 ---
        x_res = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x); x = F.relu(x + x_res)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr=edge_attr, batch=batch)

        return global_mean_pool(x, batch)   # [1, hidden_dim]


# ---- 6b. JointGAT_GCN: GAT1 + projection + GCN over PPI ----
class JointGAT_GCN(nn.Module):
    """
    Encodes all proteins via GAT1, projects to EMBED_DIM,
    then refines with 2-layer GCN over the PPI topology.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.gat     = GAT1(hidden_dim=hidden_dim)
        self.project = nn.Linear(hidden_dim, embed_dim)
        self.gcn1    = GCNConv(embed_dim, embed_dim)
        self.gcn2    = GCNConv(embed_dim, embed_dim)

    def encode_all(self, protein_graphs, ppi_edge_index):
        """Encode every protein once; returns X [N, EMBED_DIM]."""
        embeddings = []
        for g in protein_graphs:
            g   = g.to(device)
            emb = self.gat(g)            # [1, hidden_dim]
            emb = self.project(emb)      # [1, embed_dim]
            embeddings.append(emb)
        X = torch.cat(embeddings, dim=0)                      # [N, embed_dim]
        X = F.relu(self.gcn1(X, ppi_edge_index))
        X = self.gcn2(X, ppi_edge_index)                      # [N, embed_dim]
        return X

    def forward(self, protein_graphs, ppi_edge_index):
        return self.encode_all(protein_graphs, ppi_edge_index)


# ---- 6c. ComplexPredictor: aggregate member embeddings → binary score ----
class ComplexPredictor(nn.Module):
    """
    Given per-protein embeddings H [N, EMBED_DIM] and a list of
    member indices for one complex, mean-pool the members then
    score with an MLP.

    For a batch we loop over each complex (sizes vary → can't tensor-pad easily).
    """
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear( 64,   1)
        )

    def forward(self, H, complex_indices_batch):
        """
        H                    : [N, dim]  — full protein embedding matrix
        complex_indices_batch: list of lists of int (one per complex in batch)

        Returns logits [batch_size]
        """
        logits = []
        for members in complex_indices_batch:
            idx  = torch.tensor(members, dtype=torch.long, device=H.device)
            emb  = H[idx]          # [k, dim]
            pooled = emb.mean(0)   # [dim]   — mean-pool over complex members
            logits.append(self.mlp(pooled))
        return torch.cat(logits, dim=0)   # [batch_size]


# ============================================================
# CELL 7 — Instantiate models & optimiser
# ============================================================
encoder   = JointGAT_GCN(hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM).to(device)
predictor = ComplexPredictor(dim=EMBED_DIM).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Positive-weight for class imbalance (should be ~1 if balanced, adjust if not)
pos_weight = torch.tensor([1.0], device=device)


# ============================================================
# CELL 8 — Helper: encode all proteins (called once per epoch)
# ============================================================
def encode_proteins():
    """
    Run GAT1 + project + GCN over the full protein list.
    Returns H [N, EMBED_DIM] on device.
    NOTE: protein_graphs is a global list ordered by integer index.
    """
    return encoder(protein_graphs, ppi_edge_index)   # [N, EMBED_DIM]


# ============================================================
# CELL 9 — Eval helper
# ============================================================
@torch.no_grad()
def evaluate(loader, H):
    """Returns dict of metrics."""
    predictor.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        members_list = [cx for cx, _ in batch]
        labels       = torch.tensor([lbl for _, lbl in batch],
                                    dtype=torch.float, device=device)
        logits = predictor(H, members_list)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np  = torch.sigmoid(torch.tensor(logits_np)).numpy()
    preds_np  = (probs_np >= 0.5).astype(int)

    return {
        "loss"      : F.binary_cross_entropy_with_logits(
                          torch.tensor(logits_np),
                          torch.tensor(labels_np)).item(),
        "auc"       : roc_auc_score(labels_np, probs_np),
        "ap"        : average_precision_score(labels_np, probs_np),
        "acc"       : accuracy_score(labels_np, preds_np),
        "f1"        : f1_score(labels_np, preds_np, zero_division=0),
        "precision" : precision_score(labels_np, preds_np, zero_division=0),
        "recall"    : recall_score(labels_np, preds_np, zero_division=0),
    }


# ============================================================
# CELL 10 — Training loop
# ============================================================
best_val_auc  = 0.0
best_ckpt     = "best_model.pt"
history       = []

for epoch in range(1, EPOCHS + 1):
    # ---- encode all proteins (graph structure + PPI context) ----
    encoder.train(); predictor.train()
    H = encode_proteins()   # [N, EMBED_DIM]  (grad flows through encoder)

    epoch_loss = 0.0
    n_batches  = 0

    for batch in train_loader:
        optimizer.zero_grad()

        members_list = [cx  for cx,  _ in batch]
        labels       = torch.tensor([lbl for _, lbl in batch],
                                    dtype=torch.float, device=device)

        logits = predictor(H, members_list)
        loss   = F.binary_cross_entropy_with_logits(
                     logits, labels, pos_weight=pos_weight)
        loss.backward()

        # clip gradients to stabilise training
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()

    # ---- validation (no grad, re-encode fresh embeddings) ----
    with torch.no_grad():
        encoder.eval()
        H_val  = encode_proteins()

    val_metrics = evaluate(val_loader, H_val)

    avg_loss = epoch_loss / max(n_batches, 1)
    row = {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v
                                                       for k, v in val_metrics.items()}}
    history.append(row)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Ep {epoch:3d} | train_loss={avg_loss:.4f} | "
              f"val_auc={val_metrics['auc']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f} | "
              f"val_acc={val_metrics['acc']:.4f}")

    # ---- save best checkpoint ----
    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]
        torch.save({
            "encoder"  : encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "epoch"    : epoch,
            "val_auc"  : best_val_auc,
        }, best_ckpt)

print(f"\nBest val AUC: {best_val_auc:.4f}  (checkpoint: {best_ckpt})")


# ============================================================
# CELL 11 — Test evaluation
# ============================================================
ckpt = torch.load(best_ckpt, map_location=device)
encoder.load_state_dict(ckpt["encoder"])
predictor.load_state_dict(ckpt["predictor"])

with torch.no_grad():
    encoder.eval()
    H_test = encode_proteins()

test_metrics = evaluate(test_loader, H_test)
print("\n===== TEST RESULTS =====")
for k, v in test_metrics.items():
    print(f"  {k:12s}: {v:.4f}")


# ============================================================
# CELL 12 — (Optional) Inference: predict a new complex
# ============================================================
def predict_complex(member_indices, threshold=0.5):
    """
    member_indices : list of integer protein indices (from protein_index_map)
    Returns        : probability and binary prediction
    """
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        H   = encode_proteins()
        logit = predictor(H, [member_indices])   # [1]
        prob  = torch.sigmoid(logit).item()
    label = 1 if prob >= threshold else 0
    print(f"Complex {member_indices}  →  prob={prob:.4f}  pred={label}")
    return prob, label

# Example (replace indices with real ones from your data):
# predict_complex([7442, 2193, 2580])


# ============================================================
# CELL 13 — (Optional) Plot training history
# ============================================================
try:
    import matplotlib.pyplot as plt
    df_hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df_hist["epoch"], df_hist["train_loss"], label="train loss")
    axes[0].plot(df_hist["epoch"], df_hist["val_loss"],   label="val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(df_hist["epoch"], df_hist["val_auc"], label="val AUC")
    axes[1].plot(df_hist["epoch"], df_hist["val_f1"],  label="val F1")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].set_title("Val metrics")

    plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib not available — skipping plot.")

import json
import pandas as pd

# -------- FILE PATHS --------
JSON_PATH = "/Users/rajan/github/proteinComplex/DataCleaning/protein_index_map.json"
EDGE_CSV_PATH = "/Users/rajan/github/proteinComplex/complexPrediction8sem/positive_edge_1024.csv"

OUTPUT_INDEXED = "DataCleaning/positiveEdges_indexed.csv"
OUTPUT_FILTERED = "DataCleaning/positiveEdges_filtered_original.csv"

# -------- LOAD DATA --------
with open(JSON_PATH, "r") as f:
    protein_map = json.load(f)

df = pd.read_csv(EDGE_CSV_PATH)

print(f"Total edges in input: {len(df)}")

# -------- FILTER + MAP --------
filtered_original = []
indexed_edges = []
discarded_edges = []

for _, row in df.iterrows():
    p1 = row["Node1"]
    p2 = row["Node2"]
    
    if p1 in protein_map and p2 in protein_map:
        # valid edge
        filtered_original.append([p1, p2])
        
        idx1 = protein_map[p1]
        idx2 = protein_map[p2]
        indexed_edges.append([idx1, idx2])
    else:
        # ❌ missing edge → store for audit
        discarded_edges.append([p1, p2])

# -------- CREATE DATAFRAMES --------
df_filtered = pd.DataFrame(filtered_original, columns=["Node1", "Node2"])
df_indexed = pd.DataFrame(indexed_edges, columns=["Node1", "Node2"])

# -------- SAVE FILES --------
df_filtered.to_csv(OUTPUT_FILTERED, index=False)
df_indexed.to_csv(OUTPUT_INDEXED, index=False)

# -------- STATS --------
print(f"Selected valid edges: {len(df_filtered)}")
print(f"Saved → {OUTPUT_FILTERED}")
print(f"Saved → {OUTPUT_INDEXED}")

OUTPUT_DISCARDED = "DataCleaning/edges_discarded.csv"

df_discarded = pd.DataFrame(discarded_edges, columns=["Node1", "Node2"])
df_discarded.to_csv(OUTPUT_DISCARDED, index=False)

print(f"Discarded edges (missing proteins): {len(df_discarded)}")
print(f"Saved → {OUTPUT_DISCARDED}")
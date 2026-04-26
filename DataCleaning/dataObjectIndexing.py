import pickle
import json
from torch_geometric.data import Data

# -------- INPUT / OUTPUT PATHS --------
INPUT_PKL = "/Users/rajan/github/proteinComplex/complexPrediction8sem/proteinGraphs.pkl"
OUTPUT_JSON = "DataCleaning/protein_index_map.json"
OUTPUT_PKL = "DataCleaning/proteinGraphsIndexed.pkl"

# -------- LOAD ORIGINAL DATA --------
with open(INPUT_PKL, "rb") as f:
    protein_list = pickle.load(f)

print(f"Loaded {len(protein_list)} proteins")

# -------- PROCESSING --------
protein_map = {}
filtered_list = []

for idx, data in enumerate(protein_list):
    
    # Extract protein ID (remove .pdb)
    protein_id = data.name.replace(".pdb", "")
    
    # Save mapping
    protein_map[protein_id] = idx
    
    # Create new Data object with only required fields
    new_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr
    )
    
    filtered_list.append(new_data)

# -------- SAVE JSON --------
with open(OUTPUT_JSON, "w") as f:
    json.dump(protein_map, f, indent=2)

print(f"Saved JSON mapping → {OUTPUT_JSON}")

# -------- SAVE NEW PKL --------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(filtered_list, f)

print(f"Saved filtered PKL → {OUTPUT_PKL}")

# -------- DONE --------
print("Processing complete ✅")
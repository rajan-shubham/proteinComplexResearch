import json

# -------- FILE PATHS --------
COMPLEX_JSON_PATH = "/Users/rajan/github/proteinComplex/complexPrediction8sem/reformatted_filtered_complexes.json"

PROTEIN_MAP_PATH = "DataCleaning/protein_index_map.json"

OUTPUT_COMPLEXES_INDEXED = "DataCleaning/indexed_complexes.json"

# -------- LOAD FILES --------
with open(COMPLEX_JSON_PATH, "r") as f:
    complexes = json.load(f)

with open(PROTEIN_MAP_PATH, "r") as f:
    protein_map = json.load(f)

print(f"Loaded {len(complexes)} complexes")
print(f"Loaded {len(protein_map)} protein mappings")

# -------- INDEX COMPLEXES --------
indexed_complexes = []

for complex_group in complexes:
    
    # Convert each protein ID to its mapped index
    indexed_group = [
        protein_map[protein_id]
        for protein_id in complex_group
    ]
    
    indexed_complexes.append(indexed_group)

# -------- SAVE NEW JSON --------
with open(OUTPUT_COMPLEXES_INDEXED, "w") as f:
    json.dump(indexed_complexes, f, indent=2)

print(f"Saved indexed complexes → {OUTPUT_COMPLEXES_INDEXED}")

# -------- SAMPLE OUTPUT --------
print("\nSample Indexed Complexes:")
for i in range(min(5, len(indexed_complexes))):
    print(indexed_complexes[i])

print("\nProcessing complete ✅")
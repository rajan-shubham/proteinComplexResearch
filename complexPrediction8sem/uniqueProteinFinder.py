import pandas as pd
import json

# Load the CSV file
file_path = "complexPrediction8sem/positive_edge_1024.csv"   
df = pd.read_csv(file_path)

# Ensure column names are correct
df.columns = ["Node1", "Node2"]

# Combine both columns to get all proteins
all_proteins = pd.concat([df["Node1"], df["Node2"]])

# Find unique proteins
unique_proteins = sorted(all_proteins.unique())

# Print count to verify
print("Total unique proteins:", len(unique_proteins))

# Save to JSON file
output_file = "unique_proteins.json"

with open(output_file, "w") as f:
    json.dump(unique_proteins, f, indent=4)

print("Unique proteins saved to:", output_file)
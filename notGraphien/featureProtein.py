# Install required packages (for Colab use)
# pip install biopython networkx scipy nglview --quiet

# Import necessary libraries
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# Download an example AlphaFold PDB file (e.g., AlphaFold model for human insulin - AF-P01308-F1-model_v4.pdb)
pdb_id = "AF-P01308-F1-model_v4.pdb"
pdb_url = f"https://alphafold.ebi.ac.uk/files/{pdb_id}"
if not os.path.exists(pdb_id):
    urllib.request.urlretrieve(pdb_url, pdb_id)

# Parse PDB file
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", pdb_id)

# Extract C-alpha atoms and residue data
residues = []
coords = []

for model in structure:
    for chain in model:
        for res in chain:
            if is_aa(res) and "CA" in res:
                residues.append(res)
                coords.append(res["CA"].get_coord())

coords = np.array(coords)

# Compute pairwise distances
distance_matrix = squareform(pdist(coords))

# Define a dummy van der Waals interaction score matrix (normally from physico-chemical properties or ML model)
# For illustration, we'll use an inverse-square model to simulate "higher interaction" at shorter distances
interaction_scores = 1 / (distance_matrix**2 + 1e-5)  # Avoid division by zero

# Construct graph based on threshold
G = nx.Graph()
for i, res in enumerate(residues):
    aa = seq1(res.get_resname())
    G.add_node(i, resname=aa, coord=coords[i])

for i in range(len(residues)):
    for j in range(i + 1, len(residues)):
        if distance_matrix[i][j] < 5.0 and interaction_scores[i][j] > 0.04:  # Interaction score threshold
            G.add_edge(i, j, weight=interaction_scores[i][j])

# Plot graph (simplified 2D view)
pos = {i: (coords[i][0], coords[i][1]) for i in range(len(residues))}
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()})
plt.title("Protein Graph: Nodes = Amino Acids, Edges = High vdw + Short Distance")
plt.show()

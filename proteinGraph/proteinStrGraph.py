"""import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# === STEP 1: Feature dictionary for amino acids ===
aa_features = {
    'A': [8.1,  0,  1.8, 89.1],
    'C': [5.5,  0,  2.5, 121.2],
    'D': [13.0, -1, -3.5, 133.1],
    'E': [12.3, -1, -3.5, 147.1],
    'F': [5.2,  0,  2.8, 165.2],
    'G': [9.0,  0, -0.4, 75.1],
    'H': [10.4, 1, -3.2, 155.2],
    'I': [5.2,  0,  4.5, 131.2],
    'K': [11.3, 1, -3.9, 146.2],
    'L': [4.9,  0,  3.8, 131.2],
    'M': [5.7,  0,  1.9, 149.2],
    'N': [11.6, 0, -3.5, 132.1],
    'P': [8.0,  0, -1.6, 115.1],
    'Q': [10.5, 0, -3.5, 146.1],
    'R': [10.5, 1, -4.5, 174.2],
    'S': [9.2,  0, -0.8, 105.1],
    'T': [8.6,  0, -0.7, 119.1],
    'V': [5.9,  0,  4.2, 117.1],
    'W': [5.4,  0, -0.9, 204.2],
    'Y': [6.2,  0, -1.3, 181.2]
}

# === STEP 2: Build Graph from a protein string ===
def build_protein_graph(sequence):
    G = nx.Graph()
    seq = sequence.strip()

    print("sec",seq)

    for i, aa in enumerate(seq):
        print(aa)
        if aa not in aa_features:
            continue
        G.add_node(i, aa=aa, features=np.array(aa_features[aa]))

    
    
    print(G)
    nodes = list(G.nodes)
    print('nodes',nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            f1 = G.nodes[nodes[i]]['features']
            f2 = G.nodes[nodes[j]]['features']
            dist = np.linalg.norm(f1 - f2)
            G.add_edge(nodes[i], nodes[j], weight=dist)

    return G

# === STEP 3: Your input string ===
protein_string = "KGLLALALVFSLPVFAAEHWIDVRVPEQYQQEHVQGAINIPLKEVKERIATAVPDKNDTVKVYCNAGRQSGQAKEILSEMGYTHVENAGGLKDIAMPKVKG"

# === STEP 4: Build and plot the graph ===
graph = build_protein_graph(protein_string)

# Plotting
pos = nx.spring_layout(graph, seed=42)
node_labels = nx.get_node_attributes(graph, 'aa')
edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in graph.edges(data=True)}

plt.figure(figsize=(12, 10))
nx.draw(graph, pos, labels=node_labels, node_color='red', node_size=600, font_size=10)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
plt.title("Protein String Graph", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()"""


import networkx as nx
import matplotlib.pyplot as plt

# Your protein sequence
sequence = "KGLLALALVFSLPVFAAEHWIDVRVPEQYQQEHVQGAINIPLKEVKERIATAVPDKNDTVKVYCNAGRQSGQAKEILSEMGYTHVENAGGLKDIAMPKVKG"

# Create a graph
G = nx.Graph()

# Add nodes and edges
for i, aa in enumerate(sequence):
    G.add_node(i, label=aa)
    if i > 0:
        G.add_edge(i - 1, i)

# Prepare labels for display
labels = nx.get_node_attributes(G, 'label')

# Draw the graph
plt.figure(figsize=(20, 6))
pos = nx.spring_layout(G, k=0.5, iterations=100)
nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color='orange', font_size=10, font_weight='bold')
plt.title("Protein Sequence Graph: Amino Acids as Nodes", fontsize=14)
plt.axis('off')
plt.show()

# Protein Graph Constructor from AlphaFoldDB

## Project Overview

This project provides a Python script (`protein_graph_constructor.py`) that constructs 3D geometric graphs from protein structures obtained from the AlphaFold Protein Structure Database. The script takes a UniProt accession ID as input, downloads the corresponding protein structure, builds a graph representation, and visualizes it.

The graph nodes represent Alpha Carbon (CA) atoms of the amino acids. Edges are initially constructed based on peptide bonds, hydrophobic interactions, and aromatic interactions. These edges are then filtered to retain only those where the distance between the connected CA atoms is within a specified threshold (defaulting to 5.0 Ångströms).

## Key Libraries Used

*   **Graphein**: The core library used for protein graph construction, downloading PDB files from AlphaFoldDB, and generating 3D visualizations.
*   **NetworkX**: Graphein uses NetworkX as its underlying graph library. The constructed graph is a NetworkX graph object.
*   **NumPy**: Used for numerical operations, particularly for calculating distances between atomic coordinates during the edge filtering process.
*   **Plotly**: Used by Graphein to create interactive 3D visualizations of the protein structure graphs.

## Script Functionality (`protein_graph_constructor.py`)

The main script performs the following steps:

1.  **Input**: Takes a UniProt accession ID (e.g., "Q8W3K0") defined within the script.
2.  **Download PDB**: Uses `graphein.protein.utils.download_alphafold_structure` to download the PDB file for the given UniProt ID from the AlphaFoldDB. Downloaded files are stored in a local directory (default: `./alphafold_pdbs`).
3.  **Graph Construction**:
    *   **Nodes**: Each node in the graph represents the Alpha Carbon (CA) atom of an amino acid residue (`granularity='CA'`).
    *   **Edges**: Edges are added based on the following interaction types using Graphein's built-in functions:
        *   `add_peptide_bonds`: Connects adjacent amino acids in the protein sequence.
        *   `add_hydrophobic_interactions`: Identifies and adds edges for hydrophobic interactions.
        *   `add_aromatic_interactions`: Identifies and adds edges for aromatic interactions.
4.  **Edge Filtering**: After the initial graph construction, all edges are iterated through. An edge is kept only if the Euclidean distance between the 3D coordinates of the connected CA atoms is less than or equal to a `distance_threshold` (default: 5.0 Å).
5.  **Output & Visualization**:
    *   Prints information about the downloaded PDB file.
    *   Prints the number of nodes and edges in the graph before and after filtering.
    *   Prints sample data for a node and an edge.
    *   Uses `graphein.protein.visualisation.plotly_protein_structure_graph` to generate an interactive 3D plot of the final graph. Nodes are colored by residue name, and edges are colored by their `kind` (e.g., "peptide_bond", "hydrophobic"). The plot is displayed in the default web browser or a Plotly-compatible viewer.

## Setup and Execution

It is highly recommended to use a Conda environment to manage dependencies and avoid conflicts.

1.  **Create a Conda Environment**:
    If you haven't already, create a new Conda environment (e.g., named `graphein_env` with Python 3.9):
    ```bash
    conda create --name graphein_env python=3.9 -y
    ```
    *(Note: The script was developed and tested using the Anaconda Python interpreter at `/Users/mohankumar/opt/anaconda3/bin/conda` and `/Users/mohankumar/opt/anaconda3/bin/python`)*

2.  **Activate the Environment**:
    ```bash
    conda activate graphein_env
    ```

3.  **Install Dependencies**:
    The primary dependency is `graphein`, which will pull in `networkx`, `numpy`, `pandas`, `plotly`, `scipy`, and other necessary packages. Install it using pip within the activated environment:
    ```bash
    pip install graphein
    ```

4.  **Run the Script**:
    Navigate to the directory containing `protein_graph_constructor.py` and run:
    ```bash
    python protein_graph_constructor.py
    ```
    If you are not using an activated Conda environment as the default `python`, you might need to specify the full path to the Python interpreter within your Conda environment, for example:
    ```bash
    /path/to/your/conda/envs/graphein_env/bin/python protein_graph_constructor.py
    ```
    *(During development, the path `/Users/mohankumar/opt/anaconda3/envs/graphein_env/bin/python` was used).*

## File Structure

```
.
├── protein_graph_constructor.py  # The main Python script
├── alphafold_pdbs/               # Directory created by the script to store downloaded PDB files
│   └── Q8W3K0.pdb                # Example PDB file (downloaded on first run with this ID)
└── README.md                     # This file
```

## Customization

*   **Change Protein**: To process a different protein, modify the `protein_id_to_process` variable within the `if __name__ == "__main__":` block in `protein_graph_constructor.py`.
*   **Adjust Distance Threshold**: The `distance_threshold` variable in the `create_protein_graph` function call (within the `if __name__ == "__main__":` block) can be changed to alter the edge filtering criterion.
*   **Modify Interaction Types**: The `edge_funcs` list within the `create_protein_graph` function defines which types of interactions are initially considered. You can add or remove functions from Graphein's `graphein.protein.edges` modules to customize this. Remember that the subsequent distance filtering will apply to all edges generated by these functions.
*   **Visualization Parameters**: The call to `plotly_protein_structure_graph` can be further customized with other parameters supported by the function for different visual appearances (e.g., `node_size_multiplier`, `label_node_ids`).

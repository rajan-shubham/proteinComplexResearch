import os
import networkx as nx
import numpy as np # Added for distance calculation

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.utils import download_alphafold_structure
from graphein.protein.edges.distance import add_hydrophobic_interactions, add_aromatic_interactions, add_peptide_bonds
# For optional visualization:
from graphein.protein.visualisation import plotly_protein_structure_graph

def create_protein_graph(uniprot_id: str, out_dir: str = "./protein_data", distance_threshold: float = 5.0) -> nx.Graph:
    """
    Constructs a 3D geometric graph for a protein from AlphaFoldDB.

    Nodes represent amino acids (CA atoms). Edges are added if amino acids
    are involved in hydrophobic or aromatic interactions and are within the
    specified distance_threshold.

    :param uniprot_id: UniProt accession code for the protein.
    :param out_dir: Directory to download PDB file.
    :param distance_threshold: The maximum distance (in Angstroms) for
                               considering an interaction.
    :return: A NetworkX graph representing the protein structure.
             Returns an empty graph if construction fails.
    """
    print(f"Attempting to download PDB for UniProt ID: {uniprot_id} to {out_dir}")
    try:
        # Ensure the output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Download the PDB file.
        protein_pdb_path = download_alphafold_structure(
            uniprot_id, 
            out_dir=out_dir, 
            aligned_score=False
        )
        
        if not protein_pdb_path or not os.path.exists(protein_pdb_path):
            print(f"Failed to download or locate PDB file for {uniprot_id} at {protein_pdb_path}")
            return nx.Graph() # Return an empty graph

        print(f"Successfully downloaded PDB file to: {protein_pdb_path}")

    except Exception as e:
        print(f"Error during PDB download for {uniprot_id}: {e}")
        return nx.Graph()

    # Define edge construction functions.
    # These functions identify specific types of interactions (hydrophobic, aromatic)
    # and have their own distance cutoffs, which we set using functools.partial.
    # This combination addresses "distance < 5Å" and "high Van der Waals force"
    # (approximated by these interaction types).
    edge_funcs = [
        add_peptide_bonds, # Added peptide bonds
        add_hydrophobic_interactions,
        add_aromatic_interactions
    ]

    # Configure graph construction
    # Granularity 'CA' means nodes are Alpha Carbons, representing amino acids.
    config = ProteinGraphConfig(
        granularity='CA',
        edge_construction_functions=edge_funcs,
        # verbose=True # for more detailed Graphein output
    )

    print(f"Constructing graph for {protein_pdb_path} with distance threshold {distance_threshold}Å...")
    try:
        graph = construct_graph(path=protein_pdb_path, config=config)
        print(f"Initial graph constructed with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # Filter edges by distance
        if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
            print(f"Filtering edges by distance threshold: {distance_threshold}Å...")
            edges_to_remove = []
            for u, v, data in graph.edges(data=True):
                # Assuming 'CA' granularity, coords are stored in node attributes
                # Graphein typically stores coordinates as a numpy array in node_df['coords']
                # or directly as node attributes like G.nodes[u]['coords']
                # We need to ensure 'coords' attribute exists and is a numpy array
                coord_u = graph.nodes[u].get('coords')
                coord_v = graph.nodes[v].get('coords')

                if coord_u is not None and coord_v is not None:
                    # Ensure they are numpy arrays for distance calculation
                    if not isinstance(coord_u, np.ndarray):
                        coord_u = np.array(coord_u)
                    if not isinstance(coord_v, np.ndarray):
                        coord_v = np.array(coord_v)
                    
                    distance = np.linalg.norm(coord_u - coord_v)
                    if distance > distance_threshold:
                        edges_to_remove.append((u, v))
                else:
                    # If coordinates are missing, i might want to log this or handle it
                    print(f"Warning: Missing coordinates for nodes {u} or {v}. Cannot calculate distance for edge ({u}-{v}).")


            graph.remove_edges_from(edges_to_remove)
            print(f"Filtered graph: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    except Exception as e:
        print(f"Error during graph construction or edge filtering for {protein_pdb_path}: {e}")
        return nx.Graph()
        
    return graph

if __name__ == "__main__":
    # Example UniProt ID from the tutorial
    protein_id_to_process = "P15056"  # Probable disease resistance protein At1g58602
    output_directory = "./alphafold_pdbs" # Directory to store downloaded PDBs

    print(f"Processing protein: {protein_id_to_process}")
    
    # Construct the graph
    protein_graph = create_protein_graph(
        uniprot_id=protein_id_to_process,
        out_dir=output_directory,
        distance_threshold=5.0  # Angstroms
    )

    if protein_graph.number_of_nodes() > 0:
        print(f"\nGraph construction successful for {protein_id_to_process}.")
        # I can now work with the 'protein_graph' object (a NetworkX graph)
        # For example, print some information about a sample node:
        sample_node_id = list(protein_graph.nodes())[0]
        print(f"Data for sample node '{sample_node_id}': {protein_graph.nodes[sample_node_id]}")

        if protein_graph.number_of_edges() > 0:
            sample_edge = list(protein_graph.edges(data=True))[0]
            print(f"Data for sample edge: {sample_edge}")
        else:
            print("No edges were constructed based on the criteria.")
            
        print("\nAttempting to visualize the graph...")
        try:
            fig = plotly_protein_structure_graph(
                protein_graph,
                colour_nodes_by="residue_name", # Color nodes by amino acid type
                colour_edges_by="kind",         # Color edges by interaction type (peptide, hydrophobic, aromatic)
                label_node_ids=False,           # Don't label individual nodes to avoid clutter
                node_size_multiplier=2.0       # Slightly larger nodes
            )
            fig.show() # This will attempt to open the plot in a browser or viewer
            print("Visualization displayed. Check your browser or Plotly viewer.")
        except Exception as e:
            print(f"Could not generate or show visualization: {e}")
            print("Please ensure you have plotly installed and a suitable environment for displaying plots (e.g., a browser).")
    else:
        print(f"\nGraph construction failed or resulted in an empty graph for {protein_id_to_process}.")

"""
    # Example with a different protein that might have different interaction patterns
    # protein_id_to_process_2 = "P00533" # EGFR, a well-studied protein
    # print(f"\nProcessing another protein: {protein_id_to_process_2}")
    # protein_graph_2 = create_protein_graph(
    #     uniprot_id=protein_id_to_process_2,
    #     out_dir=output_directory,
    #     distance_threshold=5.0
    # )
    # if protein_graph_2.number_of_nodes() > 0:
    #     print(f"Graph for {protein_id_to_process_2} has {protein_graph_2.number_of_nodes()} nodes and {protein_graph_2.number_of_edges()} edges.")
    # else:
    #     print(f"Graph construction failed or resulted in an empty graph for {protein_id_to_process_2}.")
"""
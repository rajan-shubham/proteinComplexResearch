import json
import os

def filter_non_selected_complexes(selected_path, total_path, output_path):
    # Load the selected complexes (List of Lists)
    with open(selected_path, 'r') as f:
        selected_data = json.load(f)
    
    # Load the total complexes (Dictionary structure)
    with open(total_path, 'r') as f:
        total_dict = json.load(f)

    # Convert selected complexes into a set of sorted tuples for fast lookup
    # Sorting ensures order doesn't matter: ["A", "B"] == ["B", "A"]
    selected_set = {tuple(sorted(complex_list)) for complex_list in selected_data}

    non_selected_complexes = []

    # Iterate through the total dictionary
    # Structure: "id": ["id", [[protein_list]]]
    for key, value in total_dict.items():
        # Extract the actual list of proteins
        current_complex = value[1][0]
        
        # Check if the sorted version exists in our selected set
        if tuple(sorted(current_complex)) not in selected_set:
            non_selected_complexes.append(current_complex)

    # Save the result to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(non_selected_complexes, f, indent=2)

    print(f"Success! Found {len(non_selected_complexes)} non-selected complexes.")
    print(f"Output saved to: {output_path}")

# Configuration
SELECTED_FILE = 'complexPrediction8sem/reformatted_filtered_complexes.json'
TOTAL_FILE = 'complexPrediction8sem/DT1_Dict_hippie.json'
OUTPUT_FILE = 'complexPrediction8sem/non_selected_complexes.json'

# Run the script
if __name__ == "__main__":
    if os.path.exists(SELECTED_FILE) and os.path.exists(TOTAL_FILE):
        filter_non_selected_complexes(SELECTED_FILE, TOTAL_FILE, OUTPUT_FILE)
    else:
        print("Error: One or both input files were not found.")
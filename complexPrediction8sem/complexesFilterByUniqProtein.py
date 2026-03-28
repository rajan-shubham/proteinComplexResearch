import json

def reformat_and_filter_complexes(unique_proteins_file, hippie_file, output_file):
    try:
        # Load unique proteins
        with open(unique_proteins_file, 'r') as f:
            unique_set = set(json.load(f))
        
        # Load hippie complexes
        with open(hippie_file, 'r') as f:
            hippie_data = json.load(f)
            
        final_list = []

        for key in hippie_data:
            # Extract the actual protein IDs list
            protein_ids = hippie_data[key][1][0]
            
            # Strict filtering: all 3 must be in the unique set
            if all(protein in unique_set for protein in protein_ids):
                # We only append the inner list to our final array
                final_list.append(protein_ids)
        
        # Save as a single array of arrays
        with open(output_file, 'w') as f:
            json.dump(final_list, f, indent=4)
            
        print(f"Filtering & Reformating complete!")
        print(f"Input: {len(hippie_data)} complexes")
        print(f"Output: {len(final_list)} complexes in array-of-array format.")

    except Exception as e:
        print(f"Error: {e}")

# Run the updated script
reformat_and_filter_complexes('complexPrediction8sem/filtered_proteins.json', 'complexPrediction8sem/DT1_Dict_hippie.json', 'complexPrediction8sem/reformatted_filtered_complexes.json')

"""
def filter_complexes(unique_proteins_file, hippie_file, output_file):
    try:
        # 1. Load the unique proteins list
        with open(unique_proteins_file, 'r') as f:
            unique_list = json.load(f)
        
        # Convert to a set for high-speed lookups
        unique_set = set(unique_list)
        
        # 2. Load the hippie complexes
        with open(hippie_file, 'r') as f:
            hippie_data = json.load(f)
            
        filtered_complexes = {}

        # 3. Iterate through the complexes
        for key, value in hippie_data.items():
            # Based on your structure: value[1][0] is the list of 3 protein IDs
            # Example: ["Q96DX5", "Q93034", "Q15369"]
            protein_ids = value[1][0]
            
            # Check if ALL proteins in this complex exist in our unique_set
            if all(protein in unique_set for protein in protein_ids):
                filtered_complexes[key] = value
        
        # 4. Save the filtered results to a new file
        with open(output_file, 'w') as f:
            json.dump(filtered_complexes, f, indent=4)
            
        print(f"Filtering complete!")
        print(f"Original complexes: {len(hippie_data)}")
        print(f"Filtered complexes: {len(filtered_complexes)}")
        print(f"Saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the script
filter_complexes('complexPrediction8sem/filtered_proteins.json', 'complexPrediction8sem/DT1_Dict_hippie.json', 'complexPrediction8sem/filtered_complexes.json')
"""
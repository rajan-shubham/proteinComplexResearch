import json

# The list of 26 UniProt IDs to remove
to_remove = {
    "A0A087WTZ4", "A0A590UK80", "H7C2H4", "K7EMT4", "M0R2C6",
    "O75691", "O95613", "P04114", "P08F94", "P25391",
    "P49454", "P49908", "Q14789", "Q7Z7M0", "Q8IWI9",
    "Q8NFC6", "Q92813", "Q96DT5", "Q96JB1", "Q96L91",
    "Q9C0D9", "Q9H799", "Q9NNW7", "Q9NYQ7", "Q9Y485", "R4GMW8"
}

def filter_uniprot_ids(input_file, output_file):
    try:
        # 1. Load the original JSON file
        with open(input_file, 'r') as f:
            original_data = json.load(f)
        
        # 2. Filter the list
        # This assumes original_data is a list of IDs
        filtered_data = [protein for protein in original_data if protein not in to_remove]
        
        # 3. Save to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=4)
            
        print(f"Success! {len(original_data) - len(filtered_data)} IDs removed.")
        print(f"New file saved as: {output_file}")

    except FileNotFoundError:
        print("Error: The input file was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check your file formatting.")

# Run the function
filter_uniprot_ids('complexPrediction8sem/unique_proteins.json', 'complexPrediction8sem/filtered_proteins.json')
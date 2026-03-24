import os
import json

# 1. Your master list of 106 UniProt IDs
all_protein_ids = [
    "A0A087WTZ4",
        "A0A590UK80",
        "H7C2H4",
        "K7EMT4",
        "M0R2C6",
        "O14686",
        "O60494",
        "O60673",
        "O75592",
        "O75691",
        "O95071",
        "O95613",
        "O95714",
        "P01266",
        "P04114",
        "P08F94",
        "P20929",
        "P20930",
        "P21817",
        "P22352",
        "P24043",
        "P25054",
        "P25391",
        "P35555",
        "P36969",
        "P42858",
        "P46013",
        "P49454",
        "P49792",
        "P49908",
        "P51587",
        "P78527",
        "P98160",
        "P98161",
        "P98164",
        "Q03164",
        "Q12830",
        "Q13315",
        "Q14204",
        "Q14789",
        "Q15149",
        "Q16881",
        "Q6ZRS2",
        "Q7Z7M0",
        "Q86VQ6",
        "Q8IWI9",
        "Q8NEZ4",
        "Q8NFC6",
        "Q92736",
        "Q92813",
        "Q96DT5",
        "Q96JB1",
        "Q96L91",
        "Q96T58",
        "Q9BQE4",
        "Q9C0D9",
        "Q9H799",
        "Q9NNW7",
        "Q9NR09",
        "Q9NYQ7",
        "Q9UMN6",
        "Q9Y485",
        "Q9Y4A5",
        "R4GMW8"
]

# 2. Path to your folder containing .pdb files
pdb_folder = "/Users/rajan/github/proteinComplex/rcsb_pdbs"  # Update this if your folder has a different name
output_json = "complexPrediction8sem/missing_proteins.json"

def find_missing_proteins():
    # Get all filenames in the directory
    if not os.path.exists(pdb_folder):
        print(f"Error: The folder '{pdb_folder}' does not exist.")
        return

    downloaded_files = os.listdir(pdb_folder)
    
    # Create a string of all filenames to check against
    # This helps if the filename is 'O14686_4ERQ.pdb' instead of just 'O14686.pdb'
    all_filenames_concatenated = " ".join(downloaded_files)

    missing_proteins = []

    for prot_id in all_protein_ids:
        # Check if the ID exists anywhere in the list of filenames
        if prot_id not in all_filenames_concatenated:
            missing_proteins.append(prot_id)

    # 3. Save the results to a JSON file
    result_data = {
        "total_expected": len(all_protein_ids),
        "total_missing": len(missing_proteins),
        "missing_ids": missing_proteins
    }

    with open(output_json, 'w') as f:
        json.dump(result_data, f, indent=4)

    print(f"✅ Comparison complete!")
    print(f"📊 Total Proteins: {len(all_protein_ids)}")
    print(f"❌ Missing Proteins: {len(missing_proteins)}")
    print(f"📂 List saved to: {output_json}")

if __name__ == "__main__":
    find_missing_proteins()
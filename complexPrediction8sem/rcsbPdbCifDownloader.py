import requests
import os
import time

# List of your missing/error IDs
missing_ids = [
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

output_dir = "rcsb_pdbs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_pdb_id_from_uniprot(uniprot_id):
    """Search RCSB for PDB IDs linked to a UniProt ID."""
    url = f"https://search.rcsb.org/rcsbsearch/v2/query?json=%7B%22query%22%3A%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22{uniprot_id}%22%7D%7D%2C%22return_type%22%3A%22entry%22%7D"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if 'result_set' in results and len(results['result_set']) > 0:
            # Return the first (usually best/primary) PDB ID
            return results['result_set'][0]['identifier']
    return None

for up_id in missing_ids:
    print(f"Searching for: {up_id}...")
    pdb_id = get_pdb_id_from_uniprot(up_id)
    
    if pdb_id:
        print(f"  Found PDB ID: {pdb_id}. Downloading .pdb file...")
        # Direct download link for the .pdb format
        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_response = requests.get(download_url)
        
        if pdb_response.status_code == 200:
            with open(f"{output_dir}/{up_id}_{pdb_id}.pdb", "w") as f:
                f.write(pdb_response.text)
            print(f"  ✅ Saved as {up_id}_{pdb_id}.pdb")
        else:
            print(f"  ❌ Failed to download .pdb for {pdb_id} (Large structures may only be .cif)")
    else:
        print(f"  ❌ No experimental structure found in RCSB for {up_id}")
    
    time.sleep(0.9) # Be kind to the API
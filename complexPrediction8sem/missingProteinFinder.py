import requests
import os
import time

# Use the list of 106 IDs you provided
uniprot_ids = [
    "A0A087WTZ4",
    "A0A590UK80",
    "H7C2H4",
    "K7EMT4",
    "M0R2C6",
    "O14686",
    "O15230",
    "O43451",
    "O60229",
    "O60494",
    "O60673",
    "O75592",
    "O75691",
    "O94915",
    "O95071",
    "O95613",
    "O95714",
    "P01266",
    "P04114",
    "P04275",
    "P08F94",
    "P11532",
    "P12111",
    "P13611",
    "P15924",
    "P20929",
    "P20930",
    "P21359",
    "P21817",
    "P22105",
    "P22352",
    "P24043",
    "P25054",
    "P25391",
    "P35555",
    "P36969",
    "P42858",
    "P46013",
    "P46939",
    "P49454",
    "P49792",
    "P49908",
    "P51587",
    "P78527",
    "P98160",
    "P98161",
    "P98164",
    "Q01484",
    "Q02224",
    "Q03001",
    "Q03164",
    "Q07954",
    "Q12830",
    "Q12955",
    "Q13315",
    "Q14204",
    "Q14315",
    "Q14571",
    "Q14643",
    "Q14789",
    "Q15149",
    "Q16787",
    "Q16881",
    "Q5T011",
    "Q5T1H1",
    "Q6KC79",
    "Q6ZRI0",
    "Q6ZRS2",
    "Q7Z7M0",
    "Q86VQ6",
    "Q8IWI9",
    "Q8IZT6",
    "Q8N2C7",
    "Q8NCM8",
    "Q8NEZ4",
    "Q8NF91",
    "Q8NFC6",
    "Q8TD57",
    "Q8TDJ6",
    "Q8WXG9",
    "Q8WXH0",
    "Q8WXX0",
    "Q92736",
    "Q92813",
    "Q96DT5",
    "Q96JB1",
    "Q96L91",
    "Q96T58",
    "Q9BQE4",
    "Q9C0D9",
    "Q9H251",
    "Q9H799",
    "Q9NNW7",
    "Q9NR09",
    "Q9NYQ6",
    "Q9NYQ7",
    "Q9P225",
    "Q9P2D7",
    "Q9UFH2",
    "Q9UKN7",
    "Q9UMN6",
    "Q9UQ35",
    "Q9Y485",
    "Q9Y4A5",
    "Q9Y520",
    "R4GMW8"
] # ... add all 106 here

output_folder = "my_pdbs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for up_id in uniprot_ids:
    # 1. Ask the AlphaFold API where the file actually is
    lookup_url = f"https://alphafold.ebi.ac.uk/api/prediction/{up_id}"
    
    try:
        response = requests.get(lookup_url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                # 2. Get the real PDB URL from the metadata
                actual_pdb_url = data[0]['pdbUrl']
                
                # 3. Download the file
                pdb_file = requests.get(actual_pdb_url)
                with open(f"{output_folder}/{up_id}.pdb", "wb") as f:
                    f.write(pdb_file.content)
                print(f"✅ Success: {up_id}")
            else:
                print(f"❌ No AlphaFold model available for: {up_id}")
        else:
            print(f"⚠️ Server error ({response.status_code}) for: {up_id}")
        
        time.sleep(0.1) # Be nice to the server
    except Exception as e:
        print(f"🔥 Error with {up_id}: {e}")

print("Done!")
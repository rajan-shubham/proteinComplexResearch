import json
from collections import Counter

def audit_missing_proteins(unique_proteins_file, hippie_file):
    try:
        # Load unique proteins
        with open(unique_proteins_file, 'r') as f:
            unique_set = set(json.load(f))
        
        # Load hippie complexes
        with open(hippie_file, 'r') as f:
            hippie_data = json.load(f)
            
        missing_ids_found = []
        dropped_count = 0

        for key in hippie_data:
            protein_ids = hippie_data[key][1][0]
            
            # Find which IDs in this complex are NOT in our unique set
            missing_in_this_complex = [p for p in protein_ids if p not in unique_set]
            
            if missing_in_this_complex:
                dropped_count += 1
                missing_ids_found.extend(missing_in_this_complex)
        
        # Count frequencies of missing IDs
        missing_stats = Counter(missing_ids_found)
        
        print(f"--- Audit Results ---")
        print(f"Total complexes dropped: {dropped_count}")
        print(f"Total unique IDs missing from your list: {len(missing_stats)}")
        print(f"\nTop 10 'Problem' IDs (The ones causing the most drops):")
        print(f"{'UniProt ID':<15} | {'Complexes Lost':<15}")
        print("-" * 35)
        
        for pid, count in missing_stats.most_common(10):
            print(f"{pid:<15} | {count:<15}")

        # Optional: Save the missing IDs to a file for your reference
        with open('missing_report.json', 'w') as f:
            json.dump(dict(missing_stats.most_common()), f, indent=4)

    except Exception as e:
        print(f"Error: {e}")

# Run the audit
audit_missing_proteins('complexPrediction8sem/filtered_proteins.json', 'complexPrediction8sem/DT1_Dict_hippie.json')
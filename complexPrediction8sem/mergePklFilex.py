import pickle

# 1. Define your file paths
file_large = "complexPrediction8sem/protein_graphs.pkl"  # The file with 7419 graphs
file_small = "complexPrediction8sem/protein_graphsNew.pkl"    # The file with 80 graphs
output_file = "complexPrediction8sem/proteinGraphs.pkl"

def merge_pkl_files():
    try:
        # 2. Load the first file
        with open(file_large, 'rb') as f:
            list_1 = pickle.load(f)
        print(f"Loaded {len(list_1)} graphs from {file_large}")

        # 3. Load the second file
        with open(file_small, 'rb') as f:
            list_2 = pickle.load(f)
        print(f"Loaded {len(list_2)} graphs from {file_small}")

        # 4. Merge the lists
        merged_list = list_1 + list_2
        
        # 5. Save the merged list
        with open(output_file, 'wb') as f:
            pickle.dump(merged_list, f)
            
        print("-" * 30)
        print(f"✅ Success! Merged file saved as: {output_file}")
        print(f"📊 Total graphs in merged file: {len(merged_list)}")

    except Exception as e:
        print(f"🔥 Error during merging: {e}")

if __name__ == "__main__":
    merge_pkl_files()
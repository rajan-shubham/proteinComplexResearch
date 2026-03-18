import pickle

with open("complexPrediction8sem/protein_graphs.pkl", "rb") as f:
    data = pickle.load(f)

names = [d.name for d in data]

print("Total graphs:", len(names))
print("Unique graphs:", len(set(names)))
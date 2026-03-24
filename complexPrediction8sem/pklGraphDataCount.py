import pickle

with open("complexPrediction8sem/proteinGraphs.pkl", "rb") as f:
    data = pickle.load(f)

names = [d.name for d in data]

print("Total graphs:", len(names))
print("Unique graphs:", len(set(names)))
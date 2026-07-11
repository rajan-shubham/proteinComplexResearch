import pickle

with open("/Users/rajan/github/proteinComplex/DataCleaning/proteinGraphsIndexed.pkl", "rb") as f:
    data = pickle.load(f)

# names = [d.name for d in data]

breaker = 0
for d in data:
    print(d)
    breaker += 1
    if(breaker == 24):
        break

print("Total graphs:", len(names))
print("Unique graphs:", len(set(names)))
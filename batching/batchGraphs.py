import pickle
from torch_geometric.data import Batch

# Load saved Data objects
with open("protein_graphs.pkl", "rb") as f:
    datalist = pickle.load(f)

print(f"Loaded {len(datalist)} protein graphs")

batch_size = 5
batches = []

for i in range(0, len(datalist), batch_size):
    batch = datalist[i:i + batch_size]
    batches.append(Batch.from_data_list(batch))

print(f"Created {len(batches)} batches of size {batch_size}")

# Example usage
example_batch = batches[0]
print(example_batch)
print("x:", example_batch.x.shape)
print("edge_index:", example_batch.edge_index.shape)
print("pos:", example_batch.pos.shape)
print(example_batch.y[0:6])
print(example_batch.name[0:6])
print(example_batch.chain_ids[0:6])
print(example_batch.ptr[0:6])
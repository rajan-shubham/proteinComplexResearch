from pdb2sql import pdb2sql
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# pdb = pdb2sql('/Users/rajan/github/proteinComplex/sqlProtein/2w0o.pdb') # Horse spleen apoferritin
pdb = pdb2sql('/Users/rajan/github/proteinComplex/sqlProtein/1a3n.pdb') # DEOXY HUMAN HEMOGLOBIN
# pdb = pdb2sql('/Users/rajan/github/proteinComplex/sqlProtein/P00533.pdb') # Epidermal Growth Factor Receptor (EGFR) -> a protein involved in cell growth and differentiation

 
# Get coordinates of all alpha carbons
ca_atoms = pdb.get('x,y,z,resSeq', name='CA')
print("Sample CA atom:", ca_atoms[:5])  # Show first 5 atoms
print("Type of ca_atoms:", type(ca_atoms))
print("Type of first element:", type(ca_atoms[0]) if ca_atoms else "Empty")
print("Shape of ca_atoms:", np.array(ca_atoms).shape)

# Extract only the XYZ coordinates - fix the indexing
ca_array = np.array(ca_atoms)
print("Original ca_array shape:", ca_array.shape)

# Handle the nested structure - flatten the first dimension if needed
if ca_array.ndim == 3 and ca_array.shape[0] == 1:
    ca_array = ca_array[0]  # Remove the extra dimension
    print("After removing extra dimension:", ca_array.shape)

# Now extract only the first 3 columns (x, y, z)
coords = ca_array[:, :3]
print("Shape of coords:", coords.shape)
print("Data type of coords:", coords.dtype)
print("Sample coords:", coords[:5])

# Convert to float if needed
coords = coords.astype(float)
print("Coords after float conversion:", coords.shape, coords.dtype)

# Compute the pairwise distance matrix
dist_matrix = squareform(pdist(coords))

# Get the indices of pairs with distance < 5 Å (excluding self-comparisons)
threshold = 5.0
pairs = np.argwhere((dist_matrix < threshold) & (dist_matrix > 0))

print(f"Number of alpha carbons: {len(coords)}")
print(f"Number of pairs within {threshold}Å: {len(pairs)}")

# Plot all alpha carbon positions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

# Plot points
ax.scatter(x, y, z, color='green', s=50, alpha=0.7, label='Alpha Carbons')

# Draw edges between points < 5Å apart (limit to avoid overcrowding)
max_edges = 1000  # Limit edges for visualization
if len(pairs) > max_edges:
    print(f"Too many pairs ({len(pairs)}), showing only first {max_edges} for visualization")
    pairs = pairs[:max_edges]

for i, j in pairs:
    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='red', alpha=0.3, linewidth=0.5)

ax.set_title(f"3D Alpha Carbon Graph (< {threshold}Å)")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.legend()


plt.tight_layout()
plt.show()

# Optional: Print some statistics
print(f"\nDistance matrix shape: {dist_matrix.shape}")
print(f"Min distance: {dist_matrix[dist_matrix > 0].min():.2f}Å")
print(f"Max distance: {dist_matrix.max():.2f}Å")
print(f"Average distance: {dist_matrix[dist_matrix > 0].mean():.2f}Å")
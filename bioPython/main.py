from Bio.PDB import *
import numpy as np

parser = PDBParser()
structure = parser.get_structure("example", "/Users/rajan/github/proteinComplex/sqlProtein/1a3n.pdb")
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 print(atom)
#                 print(atom.get_name())
#                 print(atom.get_coord())




# by geting the atoms coordinates we can calculate distances between atoms and by its coodinates we can also visualize the structure in 3D
# do the above things for 1a3n.pdb file
# atoms = []
# for atom in structure.get_atoms():
#     atoms.append(atom)

# # Example: distance between first two atoms
# coord1 = atoms[0].get_coord()
# coord2 = atoms[1].get_coord()

# distance = np.linalg.norm(coord1 - coord2)
# print("Distance between atom 1 and atom 2:", distance, "Å")




# Visualizing the structure in 3D using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = [], [], []
for atom in structure.get_atoms():
    coord = atom.get_coord()
    x.append(coord[0])
    y.append(coord[1])
    z.append(coord[2])

fig = plt.figure(figsize=(10,8.5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c='blue', s=20)

ax.set_title("3D Atom Visualization of 1A3N")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()




from Bio.PDB import is_aa

ca_atoms = []
for residue in structure.get_residues():
    if is_aa(residue):  # only amino acids
        if "CA" in residue:
            ca_atoms.append(residue["CA"])

n = len(ca_atoms)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        dist = np.linalg.norm(ca_atoms[i].coord - ca_atoms[j].coord)
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
print("Distance matrix between C-alpha atoms:")
print(dist_matrix)
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# -----------------------------
#  G1 and 24-2 coordinates
# -----------------------------

G1_RIGHT = np.array([
    [-8,  26], [  8, 26],
    [-20, 20], [-12, 20], [ -4, 20], [  4, 20], [ 12, 20], [ 20, 20],
    [-20, 12], [-12, 12], [ -4, 14], [  4, 14], [ 12, 12], [ 20, 12],
    [ -8,  8], [ -2,  8], [  2,  8], [  8,  8], [ 26,  8],
    [-26,  4], [-20,  4], [-14,  4], [ -4,  4], [  4,  4], [ 22,  4],
    [ -8,  2], [ -2,  2], [  2,  2], [  8,  2],
    [  0,  0],
    [ -8, -2], [ -2, -2], [  2, -2], [  8, -2],
    [-26, -4], [-20, -4], [-14, -4], [ -4, -4], [  4, -4], [ 22, -4],
    [ -8, -8], [ -3, -8], [  3, -8], [  8, -8], [ 26, -8],
    [-20,-12], [-12,-12], [ -4,-14], [  4,-14], [ 12,-12], [ 20,-12],
    [-20,-20], [-12,-20], [ -4,-20], [  4,-20], [ 12,-20], [ 20,-20],
    [ -8,-26], [  8,-26],
], dtype=float)

VF24_2_RIGHT = np.array([
    [-9, 21], [-3, 21], [3, 21], [9, 21],
    [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
    [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
    [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [21, 3],
    [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3],[21, -3],
    [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
    [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
    [-9, -21], [-3, -21], [3, -21], [9, -21]
], dtype=float)

# -----------------------------
# KD-tree mapping
# -----------------------------
tree = cKDTree(VF24_2_RIGHT)
distances, indices = tree.query(G1_RIGHT)

# -----------------------------
# Create color palette 
# (same color for a G1 cluster and the 24-2 it maps to)
# -----------------------------

num_targets = len(VF24_2_RIGHT)
colors = plt.cm.get_cmap("tab20", num_targets)

g1_colors = np.array([colors(idx) for idx in indices])
vf_colors = np.array([colors(i) for i in range(num_targets)])

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax_g1, ax_24 = axes

# --- G1 Plot ---
ax_g1.scatter(G1_RIGHT[:,0], G1_RIGHT[:,1], c=g1_colors, s=80, edgecolor="black")
ax_g1.set_title("G1 Test Points (colored by destination)")
ax_g1.set_aspect("equal")
ax_g1.invert_yaxis()

# --- 24-2 Plot ---
ax_24.scatter(VF24_2_RIGHT[:,0], VF24_2_RIGHT[:,1], c=vf_colors, s=120, edgecolor="black")
ax_24.set_title("24-2 Points (each target has its color)")
ax_24.set_aspect("equal")
ax_24.invert_yaxis()

plt.show()

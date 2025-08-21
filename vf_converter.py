import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

grape = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
grape_vf = grape.iloc[:, -61:].values  # last 61 columns: G1 VF values
patient_ids = grape.iloc[:, 0].values
fundus_files = grape.iloc[:, 16].values

# Coordinates for G1 and 24-2
coords_g1 = np.array([
    [-23, 19], [-19, 19], [-15, 19], [-11, 19], [-7, 19],
    [-3, 19], [1, 19], [5, 19], [9, 19], [13, 19],
    [17, 19], [21, 19], [-23, 15], [-19, 15], [-15, 15],
    [-11, 15], [-7, 15], [-3, 15], [1, 15], [5, 15],
    [9, 15], [13, 15], [17, 15], [21, 15], [-23, 11],
    [-19, 11], [-15, 11], [-11, 11], [-7, 11], [-3, 11],
    [1, 11], [5, 11], [9, 11], [13, 11], [17, 11],
    [21, 11], [-23, 7], [-19, 7], [-15, 7], [-11, 7],
    [-7, 7], [-3, 7], [1, 7], [5, 7], [9, 7],
    [13, 7], [17, 7], [21, 7], [-23, 3], [-19, 3],
    [-15, 3], [-11, 3], [-7, 3], [-3, 3], [1, 3],
    [5, 3], [9, 3], [13, 3], [17, 3], [21, 3],
    [-23, -3], [-19, -3], [-15, -3], [-11, -3], [-7, -3],
    [-3, -3], [1, -3], [5, -3], [9, -3], [13, -3],
    [17, -3], [21, -3]
])
x_coords = [-21, -15, -9, -3, 3, 9, 15, 21, 100]  # 100 = dummy for unused
y_coords = [21, 18, 15, 12, 9, 6, 3, 0]
coords_242 = []

for i, y in enumerate(y_coords):
    for j, x in enumerate(x_coords[:-1]):  # ignore last col if 100
        # Skip dummy positions (100)
        coords_242.append([x, y])

coords_242 = np.array(coords_242)
x_g1_min, x_g1_max = coords_g1[:,0].min(), coords_g1[:,0].max()
y_g1_min, y_g1_max = coords_g1[:,1].min(), coords_g1[:,1].max()

# Keep only 24-2 points within G1 bounds
coords_242 = np.array([pt for pt in coords_242 if x_g1_min <= pt[0] <= x_g1_max and y_g1_min <= pt[1] <= y_g1_max])

# KDTree for nearest neighbor mapping
tree = cKDTree(coords_g1)

# For each 24-2 point, find nearest G1 point
_, mapping = tree.query(coords_242)

# Convert each GRAPE VF into 24-2 format
mapping = np.clip(mapping, 0, grape_vf.shape[1]-1)
grape_vf_242 = grape_vf[:, mapping]

# Removing last 2 blind spots
# blind_spot_indices = [x, y]  # fill with the two indices corresponding to the blind spot
# grape_vf_242 = np.delete(grape_vf_242, blind_spot_indices, axis=1)

# Saving Data
df_out = pd.DataFrame(grape_vf_242)
df_out.insert(0, "PatientID", patient_ids)
df_out.insert(1, "FundusImage", fundus_files)
df_out.replace(-1, np.nan, inplace=True)
df_out.to_csv("data/vf_tests/grape_24-2_converted.csv", index=False)

print("Saved converted 24-2 VF data to grape_24-2_converted.csv")
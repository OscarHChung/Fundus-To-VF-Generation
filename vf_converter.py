import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
import math

grape = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
grape_vf = grape.iloc[:, -61:].values  # last 61 columns: G1 VF values
patient_ids = grape.iloc[:, 0].values
laterality = grape.iloc[:, 1].values
fundus_files = grape.iloc[:, 16].values

# Coordinates for G1 and 24-2
G1_CANON = np.array([
    [-8,  26], [ 8,  26],
    [-20, 20], [-12, 20], [-4, 20], [ 4, 20], [12, 20], [20, 20],
    [-4,  14], [ 4,  14],
    [-20, 12], [-12, 12], [12, 12], [20, 12],
    [-8,   8], [-2,   8], [ 2,  8], [ 8,  8], [26,  8],
    [-26,  4], [-20,  4], [-12, 4], [-4,  4], [ 4,  4], [22,  4],
    [-8,   2], [-2,   2], [ 2,  2], [ 8,  2],
    [-8,  -2], [-2,  -2], [ 2, -2], [ 8, -2],
    [-26, -4], [-20, -4], [-14,-4], [-4, -4], [ 4, -4], [22, -4],
    [-8,  -8], [ 8,  -8], [26, -8],
    [-3,  -8], [ 3,  -8],
    [-20,-12], [-12,-12], [12,-12], [20,-12],
    [-4, -14], [ 4, -14],
    [-20,-20], [-12,-20], [-4,-20], [ 4,-20], [12,-20], [20,-20],
    [-8, -26], [ 8, -26],
    [ 0,   0],  # center
], dtype=float)

x_coords = [-21,-15,-9,-3,3,9,15,21]
y_coords = [21,18,15,12,9,6,3,0,-3,-6,-9,-12,-15,-18,-21]

coords_242 = np.array(
    [[x,y] for y in y_coords for x in x_coords if not (x==0 and y==0)],
    dtype=float
)

def polar_angle_deg(x, y):
    return math.degrees(math.atan2(y, x))

def spiral_order(coords_xy, eye="OD"):
    coords = coords_xy.copy()
    center_idx = int(np.argmin(np.hypot(coords[:,0], coords[:,1])))
    non_center_idx = [i for i in range(len(coords)) if i != center_idx]
    nonc = coords[non_center_idx]

    r = np.hypot(nonc[:,0], nonc[:,1])
    ang = np.array([polar_angle_deg(x, y) for x, y in nonc])

    if eye.upper() == "OS":
        nonc[:,0] *= -1
        r = np.hypot(nonc[:,0], nonc[:,1])
        ang = np.array([polar_angle_deg(x, y) for x, y in nonc])
        start_deg = -135
        rel = (ang - start_deg) % 360
        order_within_r = np.lexsort((rel, r))
    else:
        start_deg = -45
        rel = (ang - start_deg) % 360
        rel_cw = (360 - rel) % 360
        order_within_r = np.lexsort((rel_cw, r))

    ordered_idx = [non_center_idx[i] for i in order_within_r]
    return [center_idx] + ordered_idx

order_OD = spiral_order(G1_CANON, "OD")
order_OS = spiral_order(G1_CANON, "OS")

coords_g1_right = G1_CANON[order_OD]
coords_g1_left  = G1_CANON[order_OS]

tree_OD = cKDTree(coords_242)
tree_OS = cKDTree(coords_242)

def map_spiral_to_24_2(g1_values, eye="OD", pad=100):
    matrix = np.full((8,9), pad)
    # Slice spiral values to match 54 real points
    g1_values_real = g1_values[:len(real_indices)]
    for val, (row,col) in zip(g1_values_real, real_indices):
        matrix[row,col] = val
    return matrix


# Removing last 2 blind spots
# blind_spot_indices = [x, y]  # fill with the two indices corresponding to the blind spot
# grape_vf_242 = np.delete(grape_vf_242, blind_spot_indices, axis=1)

# Saving Data to CSV
# df_out = pd.DataFrame(grape_vf_242)
# df_out.insert(0, "PatientID", patient_ids)
# df_out.insert(1, "FundusImage", fundus_files)
# df_out.replace(-1, np.nan, inplace=True)
# df_out.to_csv("data/vf_tests/grape_24-2_converted.csv", index=False)

# print("Saved converted 24-2 VF data to grape_24-2_converted.csv")

# Saving Data to JSON
pad = 100

# Define 24-2 matrix template
template = np.full((8,9), pad)

# Define the positions of the 54 real points in the 8x9 matrix
# These are the row,col indices corresponding to the real test points
real_indices = [
    (0,3),(0,4),(0,5),(0,6),
    (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
    (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),
    (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),
    (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),
    (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),
    (6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),
    (7,3),(7,4),(7,5),(7,6)
]
coords_242_real = np.array([coords_242[i] for i in range(len(real_indices))])

output = []
for i, pid in enumerate(patient_ids):
    if pd.isna(pid):
        continue
    eye = str(laterality[i]).upper()
    if eye not in ["OD","OS"]:
        print(f"Unknown laterality {eye} for patient {pid}, skipping")
        continue

    g1_order = order_OD if eye=="OD" else order_OS
    vf_spiral = grape_vf[i, g1_order]
    vf_matrix = map_spiral_to_24_2(vf_spiral, eye)

    entry = {
        "PatientID": int(pid),
        "FundusImage": fundus_files[i],
        "Laterality": eye,
        "hvf": vf_matrix.tolist()
    }
    output.append(entry)

# Save JSON
with open("data/vf_tests/grape_24-2_matrix.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved 24-2 HVFs as 8x9 matrices in JSON")

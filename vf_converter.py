import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import json
from scipy.interpolate import griddata

grape = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
grape_vf = grape.iloc[:, -61:].values  # last 61 columns: G1 VF values
patient_ids = grape.iloc[:, 0].values
laterality = grape.iloc[:, 1].values
fundus_files = grape.iloc[:, 16].values

# Degree locations of G1 vf tests (NO BLIND SPOTS - 59 instead of 61)
G1_LOCATIONS_RIGHT = np.array([
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
G1_LOCATIONS_LEFT = np.array([
    [-8,  26], [  8, 26],
    [-20, 20], [-12, 20], [ -4, 20], [  4, 20], [ 12, 20], [ 20, 20],
    [-20, 12], [-12, 12], [ -4, 14], [  4, 14], [ 12, 12], [ 20, 12],
    [-26,  8], [ -8,  8], [ -2,  8], [  2,  8], [  8,  8],
    [-22,  4], [ -4,  4], [  4,  4], [14,  4], [20,  4], [26,  4],
    [ -8,  2], [ -2,  2], [  2,  2], [  8,  2],
    [  0,  0],
    [ -8, -2], [ -2, -2], [  2, -2], [  8, -2],
    [-22, -4], [ -4, -4], [  4, -4], [14,  -4], [20, -4], [26, -4],
    [-26, -8], [ -8, -8], [ -3, -8], [  3, -8], [  8, -8],
    [-20,-12], [-12,-12], [ -4,-14], [  4,-14], [ 12,-12], [ 20,-12],
    [-20,-20], [-12,-20], [ -4,-20], [  4,-20], [ 12,-20], [ 20,-20],
    [ -8,-26], [  8,-26],
], dtype=float)

# 24-2 VF Test Locations (NO blind spots - 52 not 54)
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
VF24_2_LEFT = np.array([
    [-9, 21], [-3, 21], [3, 21], [9, 21],
    [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
    [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],

    [-21, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
    [-21, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],

    [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
    [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
    [-9, -21], [-3, -21], [3, -21], [9, -21]
], dtype=float)


# Ordering of the G1 values in GRAPE
def spiral_order(eye):
    if eye == "OD":
        return [56, 57,
                43, 44, 45, 46, 47, 48,
                42, 27, 28, 29, 30, 49,
                16, 17, 18, 19, 58,
                55, 41, 26, 7, 8, 50,
                15, 3, 4, 20,
                0,
                14, 2, 1, 9,
                54, 40, 25, 6, 5, 31,
                13, 12, 11, 10, 51,
                39, 24, 23, 22, 21, 32,
                38, 37, 36, 35, 34, 33,
                53, 52] # right eye
    else:
        return [57, 56,
                48, 47, 46, 45, 44, 43,
                49, 30, 29, 28, 27, 42,
                58, 19, 18, 17, 16,
                50, 8, 7, 26, 41, 55,
                20, 4, 3, 15,
                0,
                9, 1, 2, 14,
                31, 5, 6, 25, 40, 54,
                51, 10, 11, 12, 13,
                32, 21, 22, 23, 24, 39,
                33, 34, 35, 36, 37, 38,
                52, 53] # left eye

# Remove 22nd and 33rd columns
mask = np.ones(grape_vf.shape[1], dtype=bool)
mask[21] = False
mask[32] = False
vf_removed = grape_vf[:, mask]

# Reorder each row based on laterality (now with 59 columns instead of 61)
reordered_vf = np.zeros_like(vf_removed)

for i, (vf_row, lat) in enumerate(zip(vf_removed, laterality)):
    order = spiral_order(lat)
    reordered_vf[i] = vf_row[order]

reordered_df = pd.DataFrame(
    reordered_vf,
    index=patient_ids,
    columns=[f"VF_{i}" for i in range(reordered_vf.shape[1])]
)
reordered_df.insert(0, "PatientID", patient_ids)
reordered_df.insert(1, "Laterality", laterality)
reordered_df.insert(2, "FundusFile", fundus_files)

reordered_df = reordered_df.iloc[1:]


# Mapping from G1 to 24-2
kd_right = cKDTree(VF24_2_RIGHT)
kd_left = cKDTree(VF24_2_LEFT)

mask_OD = np.array([
    [False, False, False,  True,  True,  True,  True, False, False],
    [False, False,  True,  True,  True,  True,  True,  True, False],
    [False,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  False,  True],
    [True,  True,  True,  True,  True,  True,  True,  False, True],
    [False, True,  True,  True,  True,  True,  True, True, True],
    [False, False, True,  True,  True,  True, True, True, False],
    [False, False, False, True,  True,  True, True, False, False]
], dtype=bool)
mask_OS = reversed_rows_arr = mask_OD[:, ::-1]
output = []

for i, pid in enumerate(patient_ids):
    if pd.isna(pid):
        continue

    eye = str(laterality[i]).upper()
    if eye not in ["OD", "OS"]:
        print(f"Unknown laterality {eye} for patient {pid}, skipping")
        continue

    # KD-Tree selection
    if eye == "OD":
        g1_points = G1_LOCATIONS_RIGHT
        kd_tree = kd_right
        mask = mask_OD
    else:
        g1_points = G1_LOCATIONS_LEFT
        kd_tree = kd_left
        mask = mask_OS

    # KD-Tree mapping
    vf_row = reordered_vf[i]
    distances, indices = kd_tree.query(g1_points)
    temp = [[] for _ in range(len(kd_tree.data))]
    for g1_val, idx in zip(vf_row, indices):
        temp[idx].append(g1_val)
    mapped_values = [np.mean(vals) if vals else np.nan for vals in temp]

    # Fill NaNs with nearest neighbor
    mapped_values = np.array(mapped_values)
    nan_mask = np.isnan(mapped_values)
    if nan_mask.any():
        from scipy.interpolate import griddata
        mapped_values[nan_mask] = griddata(
            points=kd_tree.data[~nan_mask],
            values=mapped_values[~nan_mask],
            xi=kd_tree.data[nan_mask],
            method='nearest'
        )

    # Insert values into 2D HVF matrix with padding = 100
    hvf_matrix = np.full(mask.shape, 100.0)
    hvf_matrix[mask] = mapped_values

    # Add to JSON
    entry = {
        "PatientID": int(pid),
        "FundusImage": fundus_files[i],
        "Laterality": eye,
        "hvf": hvf_matrix.tolist()
    }
    output.append(entry)

# Save JSON
with open("data/vf_tests/grape_24-2_matrix.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved 24-2 HVFs as 8x9 matrices with padding around eye in JSON")

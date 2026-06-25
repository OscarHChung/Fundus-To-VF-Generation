import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import json

# Degree locations of G1 vf tests (59 not including blind spots)
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

# 24-2 VF Test Locations (52 total not including blind spots)
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


# 24-2 valid-point mask for the right eye (8x9 grid; 52 valid of 72 cells)
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
mask_OS = mask_OD[:, ::-1]

# Module-level KD-trees over the 24-2 locations (reused by g1_to_hvf)
_KD_RIGHT = cKDTree(VF24_2_RIGHT)
_KD_LEFT = cKDTree(VF24_2_LEFT)


def g1_to_hvf(g1_61, laterality):
    """Map 61 raw G1 sensitivity values (GRAPE column order, including the two blind-spot
    columns at indices 21 and 32) to an 8x9 HVF 24-2 grid (Python lists), 100.0 at masked
    cells. This is the exact mapping the Baseline pipeline uses, factored out for reuse."""
    eye = str(laterality).strip().upper()
    eye = "OD" if eye.startswith("OD") else "OS"
    vf = np.asarray(g1_61, dtype=float)
    keep = np.ones(vf.shape[0], dtype=bool)
    keep[21] = False
    keep[32] = False
    vf = vf[keep]                       # 61 -> 59 (drop blind spots)
    vf = vf[spiral_order(eye)]          # reorder into G1_LOCATIONS order
    if eye == "OD":
        g1_pts, kd, mask = G1_LOCATIONS_RIGHT, _KD_RIGHT, mask_OD
    else:
        g1_pts, kd, mask = G1_LOCATIONS_LEFT, _KD_LEFT, mask_OS
    _, indices = kd.query(g1_pts)
    buckets = [[] for _ in range(len(kd.data))]
    for g1_val, idx in zip(vf, indices):
        buckets[idx].append(g1_val)
    mapped = np.array([np.mean(b) if b else np.nan for b in buckets])
    nan_mask = np.isnan(mapped)
    if nan_mask.any():
        mapped[nan_mask] = griddata(kd.data[~nan_mask], mapped[~nan_mask],
                                    kd.data[nan_mask], method="nearest")
    grid = np.full(mask.shape, 100.0)
    grid[mask] = mapped
    return grid.tolist()


def _convert_baseline():
    """Original behavior: convert the Baseline sheet's G1 fields to 8x9 HVF JSON.
    Requires pandas + openpyxl (only needed when run as a script)."""
    import pandas as pd
    grape = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
    grape_vf = grape.iloc[:, -61:].values
    patient_ids = grape.iloc[:, 0].values
    laterality = grape.iloc[:, 1].values
    fundus_files = grape.iloc[:, 16].values

    output = []
    for i, pid in enumerate(patient_ids):
        if pd.isna(pid):
            continue
        eye = str(laterality[i]).upper()
        if eye not in ["OD", "OS"]:
            print(f"Unknown laterality {eye} for patient {pid}, skipping")
            continue
        hvf_matrix = g1_to_hvf(grape_vf[i], eye)
        output.append({
            "PatientID": int(pid),
            "FundusImage": fundus_files[i],
            "Laterality": eye,
            "hvf": hvf_matrix,
        })

    with open("data/vf_tests/grape_new_vf_tests.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved 24-2 HVFs as 8x9 matrices with padding around eye in JSON")


if __name__ == "__main__":
    _convert_baseline()

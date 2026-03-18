import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import os
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# Load GRAPE data
grape = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
grape_vf = grape.iloc[:, -61:].values
patient_ids_excel = grape.iloc[:, 0].values
laterality_excel = grape.iloc[:, 1].values
fundus_files = grape.iloc[:, 16].values

# Load converted data
with open("data/vf_tests/grape_new_vf_tests.json", "r") as f:
    converted_data = json.load(f)

# Define G1 locations and spiral order (copied from existing files)
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

mask_OS = np.fliplr(mask_OD)

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
                53, 52]
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
                52, 53]

def get_g1_coords(eye):
    if eye.upper() == "OD":
        return G1_LOCATIONS_RIGHT
    elif eye.upper() == "OS":
        flipped = G1_LOCATIONS_RIGHT.copy()
        flipped[:,0] *= -1
        return flipped
    else:
        raise ValueError(f"Unknown eye laterality: {eye}")

def get_original_g1(pid, eye):
    idx = np.where(patient_ids_excel == pid)[0][0]
    vf_row = grape_vf[idx].astype(float)
    mask_cols = np.ones(vf_row.size, dtype=bool)
    mask_cols[21] = False
    mask_cols[32] = False
    vf_removed = vf_row[mask_cols]
    order = spiral_order(eye)
    reordered = vf_removed[order]
    coords = get_g1_coords(eye)
    return coords, reordered

# Compute MSE for each conversion
mse_list = []
for entry in converted_data:
    pid = entry['PatientID']
    eye = entry['Laterality']
    hvf_24_2 = np.array(entry['hvf'])
    
    coords_g1, vals_g1 = get_original_g1(pid, eye)
    
    if eye == 'OD':
        coords_24_2 = VF24_2_RIGHT
        mask = mask_OD
    else:
        coords_24_2 = VF24_2_LEFT
        mask = mask_OS
    
    vals_24_2 = hvf_24_2[mask]
    
    interpolated = griddata(coords_24_2, vals_24_2, coords_g1, method='linear')
    if np.any(np.isnan(interpolated)):
        interpolated = griddata(coords_24_2, vals_24_2, coords_g1, method='nearest')
    
    mse = np.mean((vals_g1 - interpolated)**2)
    mse_list.append(mse)

mse_array = np.array(mse_list)
best_idx = np.argmin(mse_array)
worst_idx = np.argmax(mse_array)
random_indices = np.random.choice(len(converted_data), 5, replace=False)

# Function to visualize old VF (G1)
def visualize_old_vf(row_idx):
    grape_local = pd.read_excel("data/vf_tests/grape_data.xlsx", sheet_name="Baseline")
    grape_vf_local = grape_local.iloc[:, -61:].values
    patient_ids_local = grape_local.iloc[:, 0].values
    laterality_local = grape_local.iloc[:, 1].values
    fundus_files_local = grape_local.iloc[:, 16].values
    
    row_idx += 1
    vf_row = grape_vf_local[row_idx].astype(float)
    pid = patient_ids_local[row_idx]
    eye = str(laterality_local[row_idx]).strip().upper()
    fundus = fundus_files_local[row_idx]
    
    mask_cols = np.ones(vf_row.size, dtype=bool)
    mask_cols[21] = False
    mask_cols[32] = False
    vf_removed = vf_row[mask_cols]
    
    order = spiral_order(eye)
    reordered = vf_removed[order]
    
    vals = np.where(reordered == 100, np.nan, reordered)
    coords = get_g1_coords(eye)
    
    plt.figure(figsize=(6,6))
    cmap = plt.cm.inferno
    sc = plt.scatter(coords[:,0], coords[:,1], c=vals, cmap=cmap, vmin=-1, vmax=30,
                     s=240, edgecolors='black', linewidth=0.5)
    plt.colorbar(sc, label="VF sensitivity (dB)")
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"GRAPE | G1 | Patient ID: {int(pid)} | Eye: {eye}", fontsize=12)
    plt.savefig(f"./temp_g1_{int(pid)}_{eye}.png", bbox_inches='tight')
    plt.close()
    
    return pid, eye

# Function to visualize new VF (24-2)
def visualize_new_vf(entry):
    pid = entry["PatientID"]
    eye = entry.get("Laterality", "NA")
    img = entry.get("FundusImage", "NA")
    hvf = np.array(entry["hvf"], dtype=float)
    
    if eye.upper() == "OS":
        flipped_hvf = hvf.copy()
        for r in range(hvf.shape[0]):
            mask_local = hvf[r] != 100
            idx = np.where(mask_local)[0]
            flipped_hvf[r, idx] = hvf[r, idx[::-1]]
        hvf = flipped_hvf
    
    hvf_masked = np.where(hvf == 100, np.nan, hvf)
    
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(hvf_masked, cmap=cmap, vmin=-1, vmax=30)
    plt.colorbar(im, label="VF sensitivity (dB)")
    plt.title(f"GRAPE | 24-2 | Patient ID: {pid} | Eye: {eye}", fontsize=12)
    plt.axis('off')
    plt.savefig(f"./temp_24_2_{pid}_{eye}.png", bbox_inches='tight')
    plt.close()
    
    return pid, eye

# Generate images
image_names = []
for i, idx in enumerate(list(random_indices) + [best_idx, worst_idx]):
    entry = converted_data[idx]
    if i < 5:
        name = f"random_{i+1}"
    elif i == 5:
        name = "best_preserved"
    else:
        name = "worst_preserved"
    image_names.append(name)
    
    # Visualize G1
    old_pid, old_eye = visualize_old_vf(idx)
    
    # Visualize 24-2
    new_pid, new_eye = visualize_new_vf(entry)
    
    # Combine
    image1 = Image.open(f"./temp_g1_{int(old_pid)}_{old_eye}.png")
    image2 = Image.open(f"./temp_24_2_{new_pid}_{new_eye}.png")
    
    min_height = min(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * min_height / image1.height), min_height))
    image2 = image2.resize((int(image2.width * min_height / image2.height), min_height))
    
    combined_width = image1.width + image2.width
    combined_image = Image.new("RGBA", (combined_width, min_height))
    
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    
    combined_image = combined_image.convert("RGB")
    combined_image.save(f"./{name}.jpg")
    
    # Clean up temp files
    os.remove(f"./temp_g1_{int(old_pid)}_{old_eye}.png")
    os.remove(f"./temp_24_2_{new_pid}_{new_eye}.png")

print("Saved 7 images: random_1.jpg to random_5.jpg, best_preserved.jpg, worst_preserved.jpg")
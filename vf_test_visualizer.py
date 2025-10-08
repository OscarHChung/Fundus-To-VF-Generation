import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import os

# Load JSON
with open("data/vf_tests/grape_new_vf_tests.json", "r") as f:
    data = json.load(f)

def print_vf(entry):
    pid = entry["PatientID"]
    eye = entry.get("Laterality", "NA")
    img = entry.get("FundusImage", "NA")
    hvf = np.array(entry["hvf"])

    # Flip horizontally for left eyes
    if eye.upper() == "OS":
        hvf = np.fliplr(hvf)

    print(f"\nPatientID: {pid}  |  Eye: {eye}  |  Fundus: {img}")
    print("-" * 50)

    # Pretty-print 8x9 matrix
    for row in hvf:
        line = "".join(f"{int(val):3d} " if val != 100 else "   ." for val in row)
        print(line)

def visualize_vf(row_id):
    entry = data[row_id]
    pid = entry["PatientID"]
    eye = entry.get("Laterality", "NA")
    img = entry.get("FundusImage", "NA")
    hvf = np.array(entry["hvf"], dtype=float)
    print(hvf)

    # Flip horizontally for left eyes
    if eye.upper() == "OS":
        flipped_hvf = hvf.copy()

        for r in range(hvf.shape[0]):
            mask = hvf[r] != 100
            # Extract indices of non-100 positions
            idx = np.where(mask)[0]
            # Flip the values among these indices
            flipped_hvf[r, idx] = hvf[r, idx[::-1]]
        
        hvf = flipped_hvf

    # Set padding (100) to NaN for white
    hvf_masked = np.where(hvf == 100, np.nan, hvf)

    # Colormap: darker = lower, brighter = higher
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')

    plt.figure(figsize=(6, 5))
    im = plt.imshow(hvf_masked, cmap=cmap, vmin=-1, vmax=30)
    plt.colorbar(im, label="VF sensitivity (dB)")
    plt.title(f"GRAPE | 24-2 | Patient ID: {pid} | Eye: {eye}", fontsize=12)
    plt.savefig(f"./data/vf_tests/visuals/GRAPE_24-2_ID_{int(pid)}_EYE_{eye}.png")
    plt.axis('off')
    # plt.show()

    return pid, eye

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

def reorder_spiral(values, eye):
    order = spiral_order(eye)
    reordered = np.zeros(len(order))
    for i, target_idx in enumerate(order):
        reordered[i] = values[target_idx]
    return reordered

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

def get_g1_coords(eye):
    if eye.upper() == "OD":
        return G1_LOCATIONS_RIGHT
    elif eye.upper() == "OS":
        # Mirror horizontally
        flipped = G1_LOCATIONS_RIGHT.copy()
        flipped[:,0] *= -1
        return flipped
    else:
        raise ValueError(f"Unknown eye laterality: {eye}")

def visualize_old_vf(row_idx, grape_xlsx="data/vf_tests/grape_data.xlsx", sheet="Baseline"):
    # Load GRAPE
    row_idx += 1
    grape = pd.read_excel(grape_xlsx, sheet_name=sheet)
    grape_vf = grape.iloc[:, -61:].values  # last 61 columns: G1 VF
    patient_ids = grape.iloc[:, 0].values
    laterality = grape.iloc[:, 1].values
    fundus_files = grape.iloc[:, 16].values

    # Select row
    vf_row = grape_vf[row_idx].astype(float)
    pid = patient_ids[row_idx]
    eye = str(laterality[row_idx]).strip().upper()
    fundus = fundus_files[row_idx]

    # Remove blind-spot columns (22nd and 33rd)
    mask_cols = np.ones(vf_row.size, dtype=bool)
    mask_cols[21] = False
    mask_cols[32] = False
    vf_removed = vf_row[mask_cols]  # length 59

    # Reorder spiral
    order = spiral_order(eye)
    if len(order) != vf_removed.size:
        raise ValueError(f"spiral_order returned length {len(order)} but vf_row has {vf_removed.size}")
    reordered = vf_removed[order]

    # Map 100 -> NaN for plotting
    vals = np.where(reordered == 100, np.nan, reordered)

    # Get coordinates
    coords = get_g1_coords(eye)

    # Plot
    plt.figure(figsize=(6,6))
    cmap = plt.cm.inferno
    sc = plt.scatter(coords[:,0], coords[:,1], c=vals, cmap=cmap, vmin=-1, vmax=30,
                     s=240, edgecolors='black', linewidth=0.5)
    plt.colorbar(sc, label="VF sensitivity (dB)")
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"GRAPE | G1 | Patient ID: {int(pid)} | Eye: {eye}", fontsize=12)
    plt.savefig(f"./data/vf_tests/visuals/GRAPE_G1_ID_{int(pid)}_EYE_{eye}.png")
    # plt.show()

    return pid, eye

# Loop over patients (print first few for demo)
#for entry in data[:10]:
#    print_vf(entry)

if __name__ == '__main__':
    fundus_id = 15
    # Old heatmap of specific entry row id
    old_pid, old_eye = visualize_old_vf(fundus_id)

    # Heatmap of a specific entry row id
    new_pid, new_eye = visualize_vf(fundus_id)

    # Save heatmaps side by side
    image1 = Image.open(f"./data/vf_tests/visuals/GRAPE_G1_ID_{int(old_pid)}_EYE_{old_eye}.png")
    image2 = Image.open(f"./data/vf_tests/visuals/GRAPE_24-2_ID_{int(new_pid)}_EYE_{new_eye}.png")

    # Determine the minimum height of the two images
    min_height = min(image1.height, image2.height)

    # Resize images to the minimum height while maintaining aspect ratio
    image1 = image1.resize((int(image1.width * min_height / image1.height), min_height))
    image2 = image2.resize((int(image2.width * min_height / image2.height), min_height))

    combined_width = image1.width + image2.width
    combined_image = Image.new("RGBA", (combined_width, min_height)) # Use RGBA for transparency

    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # Save to file
    combined_image.save(f"./data/vf_tests/visuals/COMPARISON_ID_{int(old_pid)}_EYE_{old_eye}.png")

    os.remove(f"./data/vf_tests/visuals/GRAPE_G1_ID_{int(old_pid)}_EYE_{old_eye}.png")
    os.remove(f"./data/vf_tests/visuals/GRAPE_24-2_ID_{int(new_pid)}_EYE_{new_eye}.png")

    # Or display it (if running in Jupyter/IPython)
    combined_image.show()

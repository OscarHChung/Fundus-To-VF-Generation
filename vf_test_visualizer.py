import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load JSON
with open("data/vf_tests/grape_24-2_matrix.json", "r") as f:
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

def visualize_vf(entry):
    pid = entry["PatientID"]
    eye = entry.get("Laterality", "NA")
    img = entry.get("FundusImage", "NA")
    hvf = np.array(entry["hvf"], dtype=float)

    # Flip horizontally for left eyes
    if eye.upper() == "OS":
        hvf = np.fliplr(hvf)

    # Set padding (100) to NaN for white
    hvf_masked = np.where(hvf == 100, np.nan, hvf)

    # Colormap: darker = lower, brighter = higher
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')

    plt.figure(figsize=(6, 5))
    im = plt.imshow(hvf_masked, cmap=cmap, vmin=-1, vmax=30)
    plt.colorbar(im, label="VF sensitivity (dB)")
    plt.title(f"PatientID: {pid} | Eye: {eye}", fontsize=12)
    plt.axis('off')
    plt.show()

# Loop over patients (print first few for demo)
for entry in data[:10]:
    print_vf(entry)

# Heatmap of patient
visualize_vf(data[6])

import json
import numpy as np
import matplotlib.pyplot as plt

# Load UWHVF data
with open("data/vf_tests/uwhvf_vf_tests.json") as fin:
    dat = json.load(fin)

# Define blind spot locations (row, col) in 8x9 array
BLIND_SPOTS_R = [(3, 7), (4, 7)]
BLIND_SPOTS_L = [(3, 1), (4, 1)]

def visualize_hvf(patient_id, eye='R', test_idx=0):
    """
    Visualize the raw HVF sensitivity (0-30 dB) from UWHVF dataset,
    with both blind spots correctly masked.
    
    Args:
        patient_id (str or int): Patient ID.
        eye (str): 'R' or 'L'.
        test_idx (int): Index of the HVF test to display.
    """
    patient_id = str(patient_id)
    
    entry_list = dat['data'][patient_id][eye.upper()]
    if test_idx >= len(entry_list):
        print(f"Patient {patient_id} {eye} only has {len(entry_list)} tests.")
        return
    
    hvf = np.array(entry_list[test_idx]['hvf'], dtype=float)
    
    # Flip left eye horizontally
    if eye.upper() == 'L':
        hvf = np.fliplr(hvf)
        blind_spots = BLIND_SPOTS_L
    else:
        blind_spots = BLIND_SPOTS_R
    
    # Mask padding (100) as NaN
    hvf_masked = np.where(hvf == 100, np.nan, hvf)
    
    # Mask blind spots
    for r, c in blind_spots:
        hvf_masked[r, c] = np.nan
    
    # Colormap: darker = lower, brighter = higher
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(hvf_masked, cmap=cmap, vmin=0, vmax=30)
    plt.colorbar(im, label="VF sensitivity (dB)")
    
    age = entry_list[test_idx].get('age', 'NA')
    if eye.upper() == "R":
        laterality = "OD"
    else:
        laterality = "OS"
    plt.title(f"UWHVF | 24-2 | Patient ID: {patient_id} | Eye: {laterality}", fontsize=12)
    plt.axis('off')
    plt.show()

# Print out the true value of the points:
print(np.array(dat['data']['647']['R'][0]['hvf']))
print(np.array(dat['data']['647']['L'][0]['hvf']))

# Visualize the points:
visualize_hvf(647, eye='R', test_idx=0)
visualize_hvf(647, eye='L', test_idx=0)


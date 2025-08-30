import json
import numpy as np

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

# Loop over patients (print first few for demo)
for entry in data[:10]:
    print_vf(entry)

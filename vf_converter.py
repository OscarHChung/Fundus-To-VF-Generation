import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load GRAPE Excel
grape = pd.read_excel("VF_and_clinical_information.xlsx", sheet_name="Baseline")

# Extract VF columns (last 61 columns)
grape_vf = grape.iloc[:, -61:].values  # shape (263, 61)

# Coordinates for G1 and 24-2 (example placeholders, need real coords in deg)
coords_g1 = np.array([...])   # shape (59, 2)
coords_242 = np.array([...])  # shape (54, 2)

# Build KDTree for nearest neighbor mapping
tree = cKDTree(coords_g1)

# For each 24-2 point, find nearest G1 point
_, mapping = tree.query(coords_242)  # indices of closest G1 points

# Convert each GRAPE VF into 24-2 format
grape_vf_242 = grape_vf[:, mapping]  # shape (263, 54)

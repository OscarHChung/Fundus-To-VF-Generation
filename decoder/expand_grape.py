import os
import json
import re


def expand_fundus_images(json_path, fundus_dir, output_path=None):
    """
    Updates each entry's 'FundusImage' field so it contains a list of all
    matching fundus image files in the fundus_dir.
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Determine if dict of entries or list
    is_dict = isinstance(data, dict)
    entries = data.values() if is_dict else data

    # List fundus files in folder
    available_files = os.listdir(fundus_dir)

    # Extract all basename prefixes from image filenames:
    # Matches patterns like:  "2_OD_1.jpg" → prefix "2_OD_"
    pattern = re.compile(r"(.+?_\w{2}_)")

    for entry in entries:
        img = entry.get("FundusImage", None)

        if img is None:
            continue

        # Convert any existing single string into list later
        if isinstance(img, list):
            first_img = img[0]
        else:
            first_img = img

        match = pattern.match(first_img)
        if not match:
            # Cannot extract prefix
            continue

        prefix = match.group(1)

        # Find ALL images sharing this prefix
        matched = sorted([f for f in available_files if f.startswith(prefix)])

        # Replace with list
        entry["FundusImage"] = matched

    # Save output
    if output_path is None:
        output_path = json_path  # overwrite original

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated file saved to: {output_path}")


if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    json_path = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    expand_fundus_images(json_path, fundus_dir)

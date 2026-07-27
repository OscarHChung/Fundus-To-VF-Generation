import numpy as np
import json

def convert(dataset2_path, dataset1_output_path):
    # Load uwhvf
    with open(dataset2_path, 'r') as f:
        dataset2 = json.load(f)
    
    # Extract coordinates and data
    coords = dataset2['coords']
    data_dict = dataset2['data']
    
    # Determine grid size
    max_x = max(coord['x'] for coord in coords)
    max_y = max(coord['y'] for coord in coords)
    
    # Create mapping from (x,y) to index
    coord_mapping = {(coord['x'], coord['y']): coord['index'] for coord in coords}
    
    # Convert to standardized format
    dataset1 = []
    
    for patient_id, patient_data in data_dict.items():
        # Process each eye (R for right, L for left)
        for eye_key in ['R', 'L']:
            if eye_key in patient_data:
                eye_tests = patient_data[eye_key]
                
                # Process each HVF test for this eye
                for test_index, test_data in enumerate(eye_tests):
                    # Use hvf_seq (sequence data) which matches the coords layout
                    if 'hvf_seq' in test_data:
                        hvf_seq = test_data['hvf_seq']
                        
                        # Process HVF values
                        processed_values = []
                        for val in hvf_seq:
                            if isinstance(val, str) and val.upper() == 'F':
                                processed_values.append(100.0)
                            else:
                                try:
                                    processed_values.append(float(val))
                                except (ValueError, TypeError):
                                    processed_values.append(100.0)
                        
                        # Create grid with default value 100.0
                        grid = [[100.0 for _ in range(max_y + 1)] for _ in range(max_x + 1)]
                        
                        # Fill the grid with actual values
                        for (x, y), index in coord_mapping.items():
                            if index < len(processed_values):
                                grid[x][y] = processed_values[index]
                        
                        # Convert R/L to OD/OS
                        laterality = 'OD' if eye_key == 'R' else 'OS'
                        
                        # Create patient record
                        patient_record = {
                            "PatientID": int(patient_id),
                            "FundusImage": f"{patient_id}_{laterality}_{test_index + 1}.jpg",
                            "Laterality": laterality,
                            "hvf": grid
                        }
                        
                        dataset1.append(patient_record)
                        print(f"Processed patient {patient_id}, {laterality}, test {test_index + 1}")
    
    # Save the converted dataset
    with open(dataset1_output_path, 'w') as f:
        json.dump(dataset1, f, indent=2)
    
    print(f"Conversion completed! Converted {len(dataset1)} records.")
    return dataset1

if __name__ == '__main__':
    file_path = "data/vf_tests/uwhvf_vf_tests.json"

    converted = convert("data/vf_tests/uwhvf_vf_tests.json", "data/vf_tests/uwhvf_vf_tests_standardized.json")

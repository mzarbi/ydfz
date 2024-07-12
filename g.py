from random import randint

import pandas as pd
import os

def generate_test_csv_files(num_files, num_rows, output_dir):
    """Generate test CSV files with structured data for easier testing."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create base data
    base_data = {
        'id': range(1, num_rows + 1),
        'name': [f'name_{i}' for i in range(1, num_rows + 1)],
        'value': [randint(0, 100) for i in range(1, num_rows + 1)]
    }
    base_df = pd.DataFrame(base_data)

    for i in range(num_files):
        # Duplicate every nth row to create duplicates (e.g., every 10th row)
        duplicate_indices = range(0, num_rows, 10)
        duplicates = base_df.iloc[duplicate_indices]

        # Combine base data with duplicates and shuffle
        combined_data = pd.concat([base_df, duplicates]).sample(frac=1).reset_index(drop=True)

        # Write to CSV
        file_name = os.path.join(output_dir, f'test_file_{i}.csv')
        combined_data.to_csv(file_name, index=False)
        print(f"Generated {file_name}")

# Example usage
output_directory = 'generated_test_files'
num_files_to_generate = 4  # Number of test CSV files
num_rows_per_file = 10     # Number of rows per CSV file

generate_test_csv_files(num_files_to_generate, num_rows_per_file, output_directory)

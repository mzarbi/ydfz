import hashlib
import os

# Example usage
directory_path = r'C:\Users\medzi\Desktop\bnp\ydfz\generated_test_files'
output_file = 'merged_output.csv'
num_splits = 4  # Number of smaller datasets

# Step 1: Get all files
input_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]


def remove_duplicate_lines(input_files, output_file):
    try:
        seen = set()  # A set to hold the hashes of seen lines
        with open(output_file, "w") as f_out:
            for input_file in input_files:
                if os.path.exists(input_file):
                    with open(input_file, "r") as f_in:
                        for line in f_in:
                            line_hash = hashlib.md5(line.encode()).digest()  # Hash the line
                            if line_hash not in seen:  # Check if hash is already seen
                                seen.add(line_hash)  # Add hash to the set
                                f_out.write(line)  # Write unique line to output file
                else:
                    print(f"Warning: {input_file} not found. Skipping.")
        print(f"Processing completed. Unique lines are written to {output_file}.")
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}")

# Usage
remove_duplicate_lines(input_files, output_file)

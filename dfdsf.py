import os
import time
import zipfile

# Function to create a large file
def create_large_file(filename, size_in_mb):
    with open(filename, 'wb') as f:
        f.write(os.urandom(size_in_mb * 1024 * 1024))

# Function to zip files without compression
def zip_files(files, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED) as zipf:
        for file in files:
            zipf.write(file)

# Function to read a file and measure the time taken
def read_file(filename):
    start_time = time.time()
    with open(filename, 'rb') as f:
        f.read()
    return time.time() - start_time

# Function to read a file from a zip archive and measure the time taken
def read_file_from_zip(zip_filename, file_inside_zip):
    start_time = time.time()
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        with zipf.open(file_inside_zip) as f:
            f.read()
    return time.time() - start_time

# Create two large files
file1 = 'large_file1.bin'
file2 = 'large_file2.bin'
create_large_file(file1, 500)  # 500 MB
create_large_file(file2, 500)  # 500 MB

# Zip the files without compression
zip_filename = 'archive.zip'
zip_files([file1, file2], zip_filename)

# Measure the time to read the original files
read_time_file1 = read_file(file1)
read_time_file2 = read_file(file2)

# Measure the time to read the files from the zip archive
read_time_zip_file1 = read_file_from_zip(zip_filename, file1)
read_time_zip_file2 = read_file_from_zip(zip_filename, file2)

# Print the results
print(f"Time to read {file1}: {read_time_file1} seconds")
print(f"Time to read {file2}: {read_time_file2} seconds")
print(f"Time to read {file1} from zip: {read_time_zip_file1} seconds")
print(f"Time to read {file2} from zip: {read_time_zip_file2} seconds")

# Cleanup
os.remove(file1)
os.remove(file2)
os.remove(zip_filename)

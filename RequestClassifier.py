import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define the paths
project_root = os.path.dirname(os.path.abspath(__file__))
testcases_path = os.path.join(project_root, 'testcases')
normal_path = os.path.join(project_root, 'normal')
malicious_path = os.path.join(project_root, 'malicious')

# Create the destination folders if they don't exist
os.makedirs(normal_path, exist_ok=True)
os.makedirs(malicious_path, exist_ok=True)

# Function to move files based on their extension
def move_file(file_path):
    if file_path.endswith('.white'):
        shutil.move(file_path, normal_path)
    else:
        shutil.move(file_path, malicious_path)

# Collect all file paths
file_paths = []
for root, dirs, files in os.walk(testcases_path):
    for file in files:
        file_paths.append(os.path.join(root, file))

# Use ThreadPoolExecutor to move files concurrently
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(move_file, file_paths), total=len(file_paths), desc="Processing files"))
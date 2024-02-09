import os
from pathlib import Path

def gather_py_files(root_dir, output_file):
    """
    Recursively finds all .py files in root_dir and its subdirectories,
    and concatenates their contents into a single file named output_file.
    """
    root_dir = Path(root_dir)
    output_path = root_dir / output_file
    
    with open(output_path, 'w') as outfile:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = Path(dirpath) / filename
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read() + '\n\n')
    print(f"All .py files have been concatenated into {output_path}")

# Example usage:
# Replace 'your_directory_path' with the path to the directory you want to search
root_directory = Path(os.getcwd()) / "pylol"
gather_py_files(root_directory, 'pylol.txt')
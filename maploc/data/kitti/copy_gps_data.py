import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_gps_for_split_images(src_base_dir: str, dest_base_dir: str):
    """
    Finds all GPS data files (.txt) in the source directory, and for each file,
    creates three copies in the destination directory with '_left', '_mid', and '_right'
    suffixes before the extension.
    
    For example:
    - Source: .../oxts/data/0000001234.txt
    - Destination copies:
        - .../oxts/data/0000001234_left.txt
        - .../oxts/data/0000001234_mid.txt
        - .../oxts/data/0000001234_right.txt
    """
    src_path = Path(src_base_dir)
    dest_path = Path(dest_base_dir)

    # Ensure the destination base directory exists
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # First, find all GPS files to get a total for the progress bar
    gps_files_to_process = []
    for dirpath, _, filenames in os.walk(src_path):
        # We are only interested in GPS data directories
        if 'oxts' in dirpath and 'data' in dirpath:
            for filename in filenames:
                if filename.endswith('.txt'):
                    gps_files_to_process.append(Path(dirpath) / filename)

    if not gps_files_to_process:
        print(f"No GPS (.txt) files found in 'oxts/data' subdirectories under '{src_path}'.")
        return

    suffixes = ['left', 'mid', 'right']

    # Process each found GPS file
    for src_file in tqdm(gps_files_to_process, desc="Copying GPS data for split parts"):
        # Determine the relative path to maintain the directory structure
        relative_path = src_file.relative_to(src_path)
        
        # Get the original file's stem (name without extension) and extension
        file_stem = src_file.stem
        file_ext = src_file.suffix

        # Create and copy for each suffix
        for suffix in suffixes:
            # Create the new filename, e.g., "0000001234_left.txt"
            new_filename = f"{file_stem}_{suffix}{file_ext}"
            
            # Construct the full destination path
            dest_file_path = dest_path / relative_path.with_name(new_filename)
            
            # Create the destination directory if it doesn't exist
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the source file to the new destination
            shutil.copy2(src_file, dest_file_path)

# --- Main execution ---
if __name__ == "__main__":
    # The source directory is the original KITTI dataset where GPS files are
    src_dir = "datasets/kitti"
    
    dest_dir = "datasets/kitti_split" 
    
    print(f"Copying GPS files from '{src_dir}' to '{dest_dir}' for split parts...")
    copy_gps_for_split_images(src_dir, dest_dir)
    print("GPS data copying complete.")
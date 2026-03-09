from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

def parse_image_names(path: Path):
    """
    Parses a KITTI split file, returning image names and shifts.
    Each line is expected to be: path/to/image.png [shift_x shift_y shift_z]
    """
    with open(path, "r") as fid:
        info = fid.read()
    names = []
    shifts = []
    for line in info.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        shift = parts[1:]
        
        # The name is in format 'date_drive/drive_sync/index.png'
        # We need to reconstruct it to get date, drive, and index separately
        path_parts = Path(name).parts
        date = path_parts[0]
        drive = path_parts[1]
        index = Path(path_parts[2]).stem # Filename without extension
        
        names.append((date, drive, Path(path_parts[2]).name)) # Keep original extension
        
        if len(shift) > 0:
            assert len(shift) == 3, f"Expected 3 shift values, but got {len(shift)}"
            shifts.append(np.array(shift, float))

    return names, shifts if shifts else None

# Define number of views to split the image into
num_views = 2
suffixes = ['_full', '_extracted']

# Path to the original split file
split_file_path = Path("maploc/data/kitti/origin_test1_files.txt")

# Parse names and shifts from the original file
names, shifts = parse_image_names(split_file_path)

# Prepare to write to the new panoramic split file
output_pano_file = "maploc/data/kitti/test1_files_augment_resized.txt"
with open(output_pano_file, "w") as f:
    # Iterate through each image from the original split file
    for i, (date, drive, index_with_ext) in tqdm(enumerate(names), total=len(names), desc="Splitting images..."):
        index = Path(index_with_ext).stem
        ext = Path(index_with_ext).suffix

        # Construct the full path to the image
        image_path_str = f"datasets/kitti/{date}/{drive}/image_02/data/{index_with_ext}"
        image_path = Path(image_path_str)

        if not image_path.exists():
            tqdm.write(f"Warning: Image not found at {image_path}, skipping.")
            continue
        
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            tqdm.write(f"Warning: Could not read image at {image_path}, skipping.")
            continue

        h, w, _ = image.shape
        split_w = w // 4 

        # We split the KITTI image into 4 parts and extract the 2 middle parts for reinforcement.
        # parts = [image[:, j*split_w:(j+1)*split_w] for j in range(num_views)]
        middle_part = image[:, split_w: split_w * 3]
        middle_part_resized = cv2.resize(middle_part, (w, h))
        parts = [image, middle_part_resized]

        # Process each part
        for j in range(num_views):
            part = parts[j]
            suffix = suffixes[j]
            new_index = f"{index}{suffix}{ext}"
            
            # Define the new path for the split image part
            new_image_dir = Path(f"datasets/kitti_extracted/{date}/{drive}/image_02/data/")
            os.makedirs(new_image_dir, exist_ok=True)
            new_image_path = new_image_dir / new_index
            
            # Save the new image part
            cv2.imwrite(str(new_image_path), part)
            
            # Get the corresponding shift data
            shift_data = shifts[i]
            
            # Write the new entry to the panoramic split file
            new_name_line = f"{date}/{drive}/{new_index}"
            shift_str = " ".join(map(str, shift_data))
            f.write(f"{new_name_line} {shift_str}\n")

print(f"Successfully created split images and '{output_pano_file}'")
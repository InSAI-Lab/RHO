#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build MGL dataset noise variants
Supports three types of noise: underexposure, overexposure, and motion blur
"""

import os
import glob
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

scenes = [
            "detroit",
            "berlin",
            "chicago",
            "washington",
            "sanfrancisco",
            "toulouse",
            "montrouge",
        ]
scenes = ["hasselt"]

def create_underexposed_image(image, factor=0.5):
    """
    Create underexposed image variant
    
    Parameters:
        image: Input image
        factor: Brightness reduction factor (0.0-1.0)
    """
    # Convert to HSV color space to manipulate brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    # Reduce brightness channel (V)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    # Ensure values are in valid range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    # Convert back to BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def create_overexposed_image(image, factor=2.5):
    """
    Create overexposed image variant
    
    Parameters:
        image: Input image
        factor: Brightness increase factor (>1.0)
    """
    # Convert to HSV color space to manipulate brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    # Increase brightness channel (V)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    # Ensure values are in valid range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    # Convert back to BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def create_motion_blur(image, kernel_size=10):
    """
    Create motion blur image variant
    
    Parameters:
        image: Input image
        kernel_size: Convolution kernel size, controls blur intensity
    """
    # Create motion blur convolution kernel (diagonal direction)
    kernel_diagonal = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel_diagonal, 1)
    kernel_diagonal /= kernel_size
    diagonal_mb = cv2.filter2D(image, -1, kernel_diagonal)

    return diagonal_mb

def copy_directory_structure(src_dir, dest_dir):
    """Copy directory structure from source to destination."""
    print(f"Copying directory structure: {src_dir} -> {dest_dir}")
    for dirpath, _, _ in os.walk(src_dir):
        structure = os.path.relpath(dirpath, src_dir)
        if structure == ".":
            continue
        new_dir = os.path.join(dest_dir, structure)
        os.makedirs(new_dir, exist_ok=True)

def process_image(args):
    """Process a single image and create its noise variants"""
    image_path, src_dir, dest_dir, under_factor, over_factor, blur_size = args
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Get relative path to maintain directory structure
        rel_path = os.path.relpath(image_path, src_dir)
        
        # Create and save corresponding image for each noise type
        noise_variants = {
            "MGL_under_exposure": create_underexposed_image(image, under_factor),
            "MGL_over_exposure": create_overexposed_image(image, over_factor),
            "MGL_motion_blur": create_motion_blur(image, blur_size)
        }
        
        for noise_type, noisy_image in noise_variants.items():
            target_path = os.path.join(dest_dir, noise_type, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cv2.imwrite(target_path, noisy_image)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def build_noisy_dataset(src_dir, dest_dir, num_workers=16, under_factor=0.5, over_factor=1.8, blur_size=15):
    """Build complete noise variant dataset"""
    # Create basic directory structure
    os.makedirs(dest_dir, exist_ok=True)
    for noise_type in ["MGL_under_exposure", "MGL_over_exposure", "MGL_motion_blur"]:
        os.makedirs(os.path.join(dest_dir, noise_type), exist_ok=True)
        print("Copying directory structure...")
        copy_directory_structure(src_dir, os.path.join(dest_dir, noise_type))

    for scene in scenes:
        print(f"Building noise variant dataset for {scene}")
        
        # Get all image files
        print("Finding image files...")
        image_search_path = os.path.join(src_dir, scene, 'images', '*.jpg')
        image_files = sorted(glob.glob(image_search_path, recursive=True))

        print(f"Found {len(image_files)} image files in {scene}, starting processing...")
        
        # Prepare processing parameters
        process_args = [(img, src_dir, dest_dir, under_factor, over_factor, blur_size) 
                        for img in image_files]
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_image, process_args), total=len(process_args), desc="Processing images"))
        
        # Create marker files to indicate dataset is downloaded/prepared
        for noise_type in ["MGL_under_exposure", "MGL_over_exposure", "MGL_motion_blur"]:
            with open(os.path.join(dest_dir, noise_type, ".downloaded"), "w") as f:
                f.write("This directory contains a prepared noisy variant of the MGL dataset.")
    
    print("Noise variant dataset build complete!")

def main():
    parser = argparse.ArgumentParser(description="Parameters for building MGL noisy datasets.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel processing workers")
    parser.add_argument("--under-factor", type=float, default=0.25, help="Underexposure factor (0-1)")
    parser.add_argument("--over-factor", type=float, default=2.5, help="Overexposure factor (>1)")
    parser.add_argument("--blur-size", type=int, default=10, help="Blur kernel size")
    parser.add_argument("--src-dir", type=str, required=True, help="Source directory containing original images")
    parser.add_argument("--dest-dir", type=str, required=True, help="Destination directory for noisy images")
    args = parser.parse_args()

    build_noisy_dataset(
        args.src_dir,
        args.dest_dir,
        args.workers,
        args.under_factor,
        args.over_factor,
        args.blur_size
    )

if __name__ == "__main__":
    main()
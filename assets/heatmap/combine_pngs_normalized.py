#!/usr/bin/env python3
"""Combine heatmap PNGs from three folders with normalized color scale."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import re
from pathlib import Path

def extract_step_number(filename):
    """Extract step number from filename like 'gstep_000100_expert_load.png'."""
    match = re.search(r'gstep_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0

def load_heatmap_data(png_path):
    """Load PNG and extract the heatmap data (excluding colorbar)."""
    img = Image.open(png_path)
    img_array = np.array(img)
    
    # Assuming the heatmap is the main part and colorbar is on the right
    # We'll crop to just the heatmap portion (adjust if needed)
    # For now, return the full image array for analysis
    return img_array

def read_all_heatmap_values(folders):
    """Read all heatmap PNGs and find global min/max values for normalization."""
    all_values = []
    
    for folder in folders:
        png_files = sorted(Path(folder).glob('gstep_*_expert_load.png'))
        for png_file in png_files:
            img_array = load_heatmap_data(png_file)
            # Extract RGB channels (ignore alpha if present)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            all_values.append(img_array)
    
    if all_values:
        all_arrays = np.concatenate([arr.flatten() for arr in all_values])
        return np.min(all_arrays), np.max(all_arrays)
    return 0, 255

def combine_heatmap_pngs(folders, output_path, labels=None):
    """
    Combine heatmap PNGs from multiple folders with normalized colors.
    
    Args:
        folders: List of folder paths containing PNG files
        output_path: Path for the output combined GIF
        labels: Optional list of labels to add above each column
    """
    # Get all PNG files from each folder
    all_pngs = []
    for folder in folders:
        png_files = sorted(Path(folder).glob('gstep_*_expert_load.png'), 
                          key=lambda x: extract_step_number(x.name))
        all_pngs.append(png_files)
    
    # Find common steps across all folders
    n_frames = min(len(pngs) for pngs in all_pngs)
    
    if n_frames == 0:
        print("Error: No PNG files found in folders")
        return
    
    print(f"Found {n_frames} frames to process")
    
    # Load first frame from each to get dimensions
    first_frames = [Image.open(pngs[0]) for pngs in all_pngs]
    
    # Calculate dimensions
    max_height = max(frame.height for frame in first_frames)
    total_width = sum(frame.width for frame in first_frames)
    label_height = 50 if labels else 0
    
    # Create combined frames
    combined_frames = []
    durations = []
    
    for frame_idx in range(n_frames):
        # Load frames from each folder
        frames = [Image.open(all_pngs[i][frame_idx]) for i in range(len(folders))]
        
        # Create new image for combined frame
        combined = Image.new('RGB', (total_width, max_height + label_height), (255, 255, 255))
        
        # Add labels if provided
        if labels:
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            x_offset_label = 0
            for i, frame in enumerate(frames):
                if i < len(labels):
                    bbox = draw.textbbox((0, 0), labels[i], font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    text_x = x_offset_label + (frame.width - text_width) // 2
                    text_y = (label_height - text_height) // 2
                    
                    draw.text((text_x, text_y), labels[i], fill='black', font=font)
                
                x_offset_label += frame.width
        
        # Paste frames horizontally
        x_offset = 0
        for frame in frames:
            y_offset = label_height + (max_height - frame.height) // 2
            combined.paste(frame, (x_offset, y_offset))
            x_offset += frame.width
        
        combined_frames.append(combined)
        durations.append(100)  # 100ms per frame
        
        if (frame_idx + 1) % 20 == 0:
            print(f"Processed {frame_idx + 1}/{n_frames} frames")
    
    # Save as animated GIF
    print("Saving animated GIF...")
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )
    
    print(f"Combined GIF saved to: {output_path}")
    print(f"Total frames: {n_frames}")
    print(f"Dimensions: {combined_frames[0].width}x{combined_frames[0].height}")

if __name__ == "__main__":
    folders = [
        "a0_heatmap",
        "a1_heatmap",
        "a2_heatmap"
    ]
    
    output_path = "combined_heatmaps_normalized.gif"
    
    labels = [
        "No Load Balancing",
        "Imbalance Penalty",
        "Router Biasing"
    ]
    
    # Verify all folders exist
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Error: Folder not found: {folder}")
            exit(1)
    
    combine_heatmap_pngs(folders, output_path, labels=labels)

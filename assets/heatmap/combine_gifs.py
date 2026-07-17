#!/usr/bin/env python3
"""Combine three animated GIFs horizontally into a single animated GIF."""

from PIL import Image, ImageDraw, ImageFont
import os

def combine_animated_gifs(gif_paths, output_path, labels=None):
    """
    Combine multiple animated GIFs horizontally.
    
    Args:
        gif_paths: List of paths to input GIF files
        output_path: Path for the output combined GIF
        labels: Optional list of labels to add above each GIF
    """
    # Open all GIF files
    gifs = [Image.open(path) for path in gif_paths]
    
    # Get the number of frames (use minimum to handle different lengths)
    n_frames = min(gif.n_frames for gif in gifs)
    
    # Get durations from the first GIF
    durations = []
    for frame_idx in range(n_frames):
        gifs[0].seek(frame_idx)
        durations.append(gifs[0].info.get('duration', 100))
    
    # Create combined frames
    combined_frames = []
    
    for frame_idx in range(n_frames):
        # Seek to the current frame in each GIF
        frames = []
        for gif in gifs:
            gif.seek(frame_idx)
            frames.append(gif.convert('RGBA'))
        
        # Calculate dimensions
        max_height = max(frame.height for frame in frames)
        total_width = sum(frame.width for frame in frames)
        
        # Add extra height for labels if provided
        label_height = 50 if labels else 0
        
        # Create new image for combined frame
        combined = Image.new('RGBA', (total_width, max_height + label_height), (255, 255, 255, 255))
        
        # Add labels if provided
        if labels:
            draw = ImageDraw.Draw(combined)
            try:
                # Try to use a nice font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            x_offset_label = 0
            for i, frame in enumerate(frames):
                if i < len(labels):
                    # Get text size
                    bbox = draw.textbbox((0, 0), labels[i], font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Center text above each frame
                    text_x = x_offset_label + (frame.width - text_width) // 2
                    text_y = (label_height - text_height) // 2
                    
                    # Draw text with black color
                    draw.text((text_x, text_y), labels[i], fill='black', font=font)
                
                x_offset_label += frame.width
        
        # Paste frames horizontally
        x_offset = 0
        for frame in frames:
            # Center vertically if needed (below the label)
            y_offset = label_height + (max_height - frame.height) // 2
            combined.paste(frame, (x_offset, y_offset))
            x_offset += frame.width
        
        combined_frames.append(combined.convert('RGB'))
    
    # Save as animated GIF
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )
    
    # Close all GIF files
    for gif in gifs:
        gif.close()
    
    print(f"Combined GIF saved to: {output_path}")
    print(f"Total frames: {n_frames}")
    print(f"Dimensions: {combined_frames[0].width}x{combined_frames[0].height}")

if __name__ == "__main__":
    gif_paths = [
        "a0_heatmap/a0_heatmaps.gif",
        "a1_heatmap/a1_heatmaps.gif",
        "a2_heatmap/a2_heatmaps.gif"
    ]
    
    output_path = "combined_heatmaps.gif"
    
    labels = [
        "No Load Balancing",
        "Imbalance Penalty",
        "Router Biasing"
    ]
    
    # Verify all input files exist
    for path in gif_paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            exit(1)
    
    combine_animated_gifs(gif_paths, output_path, labels=labels)

#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import re

def generate_gradient_map(H, W, map_type):
    """Generate the original gradient map that we're trying to mimic
    Uses the EXACT same function as in src/train_probes_online.py
    Returns the gradient as a numpy array (converted from torch tensor)"""
    if map_type not in ["Horizontal", "Vertical", "Gaussian"]:
        raise ValueError(f"{map_type} not supported")

    if map_type == "Horizontal":
        gradient = torch.linspace(0, 1, steps=W).repeat(H, 1)
    elif map_type == "Vertical":
        gradient = torch.linspace(0, 1, steps=H).view(H, 1).repeat(1, W)
    elif map_type == "Gaussian":
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        xv, yv = np.meshgrid(x, y)
        sigma = 0.5
        gauss = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        gradient = torch.tensor(gauss, dtype=torch.float32)

    # Convert to numpy (original returns torch tensor, but we need numpy for image conversion)
    return gradient.numpy()

def gradient_map_to_image(gradient, cmap_name="viridis"):
    """Convert gradient map to RGB image using the same colormap as saved images"""
    # Normalize gradient to [0, 1] range
    gradient_normalized = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    
    # Apply colormap (same as plt.imsave with cmap="viridis")
    # Use newer matplotlib API to avoid deprecation warning
    try:
        # For matplotlib >= 3.7
        from matplotlib import colormaps
        colormap = colormaps[cmap_name]
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        colormap = cm.get_cmap(cmap_name)
    
    gradient_rgb = colormap(gradient_normalized)
    
    # Convert to uint8 (0-255) and remove alpha channel
    gradient_rgb = (gradient_rgb[:, :, :3] * 255).astype(np.uint8)
    
    return gradient_rgb

def find_layers(images_dir, timestep):
    """Find all available layers for a given timestep"""
    layers = []
    timestep_dir = Path(images_dir) / f"timestep_{timestep}"
    
    if not timestep_dir.exists():
        raise ValueError(f"Timestep directory not found: {timestep_dir}")
    
    for item in timestep_dir.iterdir():
        if item.is_dir() and item.name.startswith('layer_'):
            match = re.search(r'layer_(\d+)', item.name)
            if match:
                layers.append(int(match.group(1)))
    
    return sorted(layers)  # Ascending order (layer_00, layer_01, ...)

def load_image(image_path):
    """Load and return image as numpy array"""
    if not image_path.exists():
        return None
    img = Image.open(image_path)
    return np.array(img)

def create_layer_grid(images_dir, example_i, kernel_size, timestep, grad_type,
                     images_per_row=10, output_path=None, layer_step=1, 
                     include_last_layer=False, layer_start=None, layer_end=None,
                     all_gradient_types=False):
    """
    Create a grid of images showing layer representations for a given timestep
    
    Args:
        images_dir: Base directory containing timestep_* folders
        example_i: Example index (e.g., 0, 1, 2...)
        kernel_size: Kernel size (e.g., 1)
        timestep: Fixed timestep to use
        grad_type: Gradient type (e.g., "Gaussian", "Vertical", "Horizontal") - used if all_gradient_types=False
        images_per_row: Number of images per row (number of layers per row)
        output_path: Path to save the output image
        layer_step: Step size for sampling layers (e.g., 5 means indices 0, 5, 10, 15...)
        include_last_layer: If True, include the last layer even if not in step sequence
        layer_start: Start layer index (inclusive, for range mode)
        layer_end: End layer index (inclusive, for range mode)
        all_gradient_types: If True, create one row per gradient type (Gaussian, Vertical, Horizontal)
    """
    all_layers = find_layers(images_dir, timestep)
    
    if len(all_layers) == 0:
        raise ValueError(f"No layer directories found for timestep {timestep} in {images_dir}")
    
    print(f"Found {len(all_layers)} layers for timestep {timestep}: {all_layers}")
    
    # Filter layers based on mode
    if layer_start is not None and layer_end is not None:
        # Range mode
        layers = [l for l in all_layers if layer_start <= l <= layer_end]
        print(f"Using range mode: layers {layer_start} to {layer_end}: {layers}")
    elif layer_step > 1:
        # Sampling mode
        sampled_indices = list(range(0, len(all_layers), layer_step))
        layers = [all_layers[i] for i in sampled_indices]
        
        # Add last layer if flag is set and it's not already included
        if include_last_layer and len(all_layers) > 0:
            last_layer = all_layers[-1]
            if last_layer not in layers:
                layers.append(last_layer)
        
        print(f"Sampled {len(layers)} layers with step={layer_step}: {layers}")
    else:
        # All layers
        layers = all_layers
    
    if len(layers) == 0:
        raise ValueError(f"No layers to display")
    
    # Determine which gradient types to show
    if all_gradient_types:
        gradient_types = ["Gaussian", "Vertical", "Horizontal"]
    else:
        gradient_types = [grad_type]
    
    # Images are saved at 512x512 (as per save_eval_image function)
    H, W = 512, 512
    
    # Build grid: each row is a different gradient type
    all_row_images = []
    all_row_labels = []
    
    for grad_type_row in gradient_types:
        # Generate target gradient map for this row
        target_gradient = generate_gradient_map(H, W, grad_type_row)
        gradient_image = gradient_map_to_image(target_gradient, cmap_name="viridis")
        
        # Start row with target map
        row_images = [gradient_image]
        row_labels = ["Target"]
        
        # Load all layer images for this gradient type
        for layer in layers:
            image_path = Path(images_dir) / f"timestep_{timestep}" / f"layer_{layer:02d}" / \
                         f"example_{example_i}_kernel_{kernel_size}_grad_{grad_type_row}.png"
            
            img = load_image(image_path)
            if img is not None:
                row_images.append(img)
                row_labels.append(f"L{layer}")
            else:
                print(f"Warning: Image not found: {image_path}")
        
        if len(row_images) == 1:  # Only target map, no layer images
            print(f"Warning: No layer images found for {grad_type_row}")
        
        all_row_images.append(row_images)
        all_row_labels.append(row_labels)
    
    # Calculate grid dimensions
    # Each row has: 1 target + len(layers) images
    num_cols_per_row = 1 + len(layers)  # Target + layers
    num_gradient_types = len(gradient_types)
    
    # For single gradient type, respect images_per_row; for multiple types, use all columns
    if num_gradient_types == 1:
        # Single gradient type: use images_per_row to control layout
        actual_cols = min(num_cols_per_row, images_per_row)
        # If we have more images than images_per_row, we'll wrap to multiple rows
        if num_cols_per_row > images_per_row:
            num_rows = (num_cols_per_row + images_per_row - 1) // images_per_row
        else:
            num_rows = 1
    else:
        # Multiple gradient types: use all columns (one row per type)
        actual_cols = num_cols_per_row
        num_rows = num_gradient_types
    
    # Create figure (larger images, better use of space)
    # Use actual number of columns for width calculation to avoid empty space
    fig = plt.figure(figsize=(actual_cols * 1.5, num_rows * 1.5))
    gs = GridSpec(num_rows, actual_cols, figure=fig, 
                  hspace=0.2, wspace=0.2,
                  left=0.0, right=1.0, top=1.0, bottom=0.0)
    
    # Plot images: each row is a different gradient type (or wrapped rows for single type)
    for grad_type_idx, (row_images, row_labels) in enumerate(zip(all_row_images, all_row_labels)):
        grad_type_row = gradient_types[grad_type_idx]
        
        if num_gradient_types == 1 and num_rows > 1:
            # Single gradient type with wrapping (multiple rows)
            for img_idx, (img, label) in enumerate(zip(row_images, row_labels)):
                row = img_idx // actual_cols
                col = img_idx % actual_cols
                
                if row >= num_rows:
                    break  # Skip if we exceed row limit
                
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img)
                ax.axis('off')
                
                # Add label at the top (larger font for formal paper)
                ax.text(0.5, 1.02, label, transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=16)
        else:
            # No wrapping needed (single row) or multiple gradient types
            for col_idx, (img, label) in enumerate(zip(row_images, row_labels)):
                if col_idx >= actual_cols:
                    break  # Skip if we exceed column limit
                
                ax = fig.add_subplot(gs[grad_type_idx, col_idx])
                ax.imshow(img)
                ax.axis('off')
                
                # Add label at the top (larger font for formal paper)
                # For first column, add gradient type label; for others, add layer
                if col_idx == 0 and all_gradient_types:
                    # Show only gradient type for first column when showing all types (no "Target")
                    ax.text(0.5, 1.02, grad_type_row, transform=ax.transAxes, 
                            ha='center', va='bottom', fontsize=18)
                else:
                    ax.text(0.5, 1.02, label, transform=ax.transAxes, 
                            ha='center', va='bottom', fontsize=16)
    
    # Hide unused subplots (if any)
    # Calculate total used cells
    if num_gradient_types == 1 and num_rows > 1:
        # Single gradient type with wrapping
        total_used = len(all_row_images[0])
    else:
        # Multiple gradient types or single row
        total_used = sum(len(row) for row in all_row_images)
    
    for idx in range(total_used, num_rows * actual_cols):
        row = idx // actual_cols
        col = idx % actual_cols
        if row < num_rows and col < actual_cols:
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    # Save figure (no title)
    if output_path is None:
        if all_gradient_types:
            output_path = Path(images_dir) / f"layer_grid_timestep_{timestep}_example_{example_i}_kernel_{kernel_size}_all_grads.png"
        else:
            output_path = Path(images_dir) / f"layer_grid_timestep_{timestep}_example_{example_i}_kernel_{kernel_size}_grad_{grad_type}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid image to: {output_path}")
    
    # Save PDF (same filename but with .pdf extension)
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved grid PDF to: {pdf_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Create a grid of images showing layer representations for a given timestep"
    )
    parser.add_argument("--images_dir", type=str, 
                       default="probing/images_ln_conv_sana",
                       help="Base directory containing timestep_* folders")
    parser.add_argument("--example_i", type=int, required=True,
                       help="Example index (e.g., 0, 1, 2...)")
    parser.add_argument("--kernel_size", type=int, required=True,
                       help="Kernel size (e.g., 1)")
    parser.add_argument("--timestep", type=int, required=True,
                       help="Fixed timestep to use (e.g., 249)")
    parser.add_argument("--grad_type", type=str, required=False,
                       choices=["Gaussian", "Vertical", "Horizontal"],
                       help="Gradient type (required if --all_gradient_types is not set)")
    parser.add_argument("--images_per_row", type=int, default=10,
                       help="Number of images per row (number of layers per row)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for the grid image (default: auto-generated)")
    parser.add_argument("--layer_step", type=int, default=1,
                       help="Step size for sampling layers (e.g., 5 means indices 0, 5, 10, 15...). Default: 1 (all layers)")
    parser.add_argument("--include_last_layer", action="store_true",
                       help="Include the last layer even if not in step sequence")
    parser.add_argument("--layer_start", type=int, default=None,
                       help="Start layer index for range mode (inclusive)")
    parser.add_argument("--layer_end", type=int, default=None,
                       help="End layer index for range mode (inclusive)")
    parser.add_argument("--all_gradient_types", action="store_true",
                       help="Create one row per gradient type (Gaussian, Vertical, Horizontal)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_gradient_types and args.grad_type is None:
        parser.error("--grad_type is required when --all_gradient_types is not set")
    
    if args.layer_start is not None and args.layer_end is None:
        parser.error("--layer_end is required when --layer_start is provided")
    if args.layer_end is not None and args.layer_start is None:
        parser.error("--layer_start is required when --layer_end is provided")
    
    if args.all_gradient_types:
        grad_type = "Gaussian"  # Dummy value, not used when all_gradient_types=True
    else:
        grad_type = args.grad_type
    
    create_layer_grid(
        images_dir=args.images_dir,
        example_i=args.example_i,
        kernel_size=args.kernel_size,
        timestep=args.timestep,
        grad_type=grad_type,
        images_per_row=args.images_per_row,
        output_path=args.output_path,
        layer_step=args.layer_step,
        include_last_layer=args.include_last_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        all_gradient_types=args.all_gradient_types
    )

if __name__ == "__main__":
    main()


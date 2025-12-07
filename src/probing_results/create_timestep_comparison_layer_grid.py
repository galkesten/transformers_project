#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def create_timestep_comparison_grid(images_dir, example_i, kernel_size, timesteps, grad_type,
                                   output_path=None, layer_step=1,
                                   include_last_layer=False, layer_start=None, layer_end=None):
    """
    Create a grid comparing layers across multiple timesteps
    Each row is a timestep, each column is a layer
    No target gradient map - just comparing layers across timesteps
    
    Args:
        images_dir: Base directory containing timestep_* folders
        example_i: Example index (e.g., 0, 1, 2...)
        kernel_size: Kernel size (e.g., 1)
        timesteps: List of timesteps to compare (each becomes a row)
        grad_type: Gradient type (e.g., "Gaussian", "Vertical", "Horizontal")
        output_path: Path to save the output image
        layer_step: Step size for sampling layers
        include_last_layer: If True, include the last layer even if not in step sequence
        layer_start: Start layer index (inclusive, for range mode)
        layer_end: End layer index (inclusive, for range mode)
    """
    # Use first timestep to find available layers (assuming all timesteps have same layers)
    all_layers = find_layers(images_dir, timesteps[0])
    
    if len(all_layers) == 0:
        raise ValueError(f"No layer directories found for timestep {timesteps[0]} in {images_dir}")
    
    print(f"Found {len(all_layers)} layers: {all_layers}")
    
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
    
    # Generate target gradient map (images are saved at 512x512)
    H, W = 512, 512
    target_gradient = generate_gradient_map(H, W, grad_type)
    target_image = gradient_map_to_image(target_gradient, cmap_name="viridis")
    
    # Build grid: each row is a different timestep
    all_row_images = []
    all_row_labels = []
    
    for timestep in timesteps:
        row_images = []
        row_labels = []
        
        # Start row with target gradient map
        row_images.append(target_image)
        row_labels.append("Target")
        
        # Load all layer images for this timestep
        for layer in layers:
            image_path = Path(images_dir) / f"timestep_{timestep}" / f"layer_{layer:02d}" / \
                         f"example_{example_i}_kernel_{kernel_size}_grad_{grad_type}.png"
            
            img = load_image(image_path)
            if img is not None:
                row_images.append(img)
                row_labels.append(f"L{layer}")
            else:
                print(f"Warning: Image not found: {image_path}")
        
        if len(row_images) == 1:  # Only target map, no layer images
            print(f"Warning: No layer images found for timestep {timestep}")
        
        all_row_images.append(row_images)
        all_row_labels.append((f"t={timestep}", row_labels))  # Store timestep label separately
    
    # Calculate grid dimensions
    # Each row has: 1 target + len(layers) images
    num_cols = 1 + len(layers)  # Target + layers
    num_rows = len(timesteps)  # Each row is a timestep
    
    # Create figure
    fig = plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
    gs = GridSpec(num_rows, num_cols, figure=fig, 
                  hspace=0.2, wspace=0.2,
                  left=0.0, right=1.0, top=1.0, bottom=0.0)
    
    # Plot images: each row is a different timestep, each column is a layer (first column is target)
    for timestep_idx, (row_images, label_data) in enumerate(zip(all_row_images, all_row_labels)):
        timestep_label, row_labels = label_data  # "t=999" and layer labels
        
        for col_idx, (img, label) in enumerate(zip(row_images, row_labels)):
            ax = fig.add_subplot(gs[timestep_idx, col_idx])
            ax.imshow(img)
            ax.axis('off')
            
            # Add label at the top: show layer label with timestep in parentheses
            # For target, just show "Target", for layers show "L0 (t=999)" format
            if label == "Target":
                combined_label = "Target"
            else:
                combined_label = f"{label} ({timestep_label})"
            ax.text(0.5, 1.02, combined_label, transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=16)
    
    # Hide unused subplots (if any)
    # Each row should have exactly len(layers) images, so this shouldn't be needed
    # But keep it as a safety check in case some images are missing
    for row in range(num_rows):
        for col in range(num_cols):
            if col >= len(all_row_images[row]):
                ax = fig.add_subplot(gs[row, col])
                ax.axis('off')
    
    # Save figure
    if output_path is None:
        timesteps_str = "_".join(map(str, timesteps))
        output_path = Path(images_dir) / f"timestep_comparison_layers_{timesteps_str}_example_{example_i}_kernel_{kernel_size}_grad_{grad_type}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid image to: {output_path}")
    
    # Save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved grid PDF to: {pdf_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Create a grid comparing layers across multiple timesteps (each row is a timestep, each column is a layer)"
    )
    parser.add_argument("--images_dir", type=str, 
                       default="probing/images_ln_conv_sana",
                       help="Base directory containing timestep_* folders")
    parser.add_argument("--example_i", type=int, required=True,
                       help="Example index (e.g., 0, 1, 2...)")
    parser.add_argument("--kernel_size", type=int, required=True,
                       help="Kernel size (e.g., 1)")
    parser.add_argument("--timesteps", type=int, nargs='+', required=True,
                       help="List of timesteps to compare (e.g., 999 749 136). Each timestep becomes a row")
    parser.add_argument("--grad_type", type=str, required=True,
                       choices=["Gaussian", "Vertical", "Horizontal"],
                       help="Gradient type")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for the grid image (default: auto-generated)")
    parser.add_argument("--layer_step", type=int, default=1,
                       help="Step size for sampling layers (e.g., 2 means indices 0, 2, 4, 6...). Default: 1 (all layers)")
    parser.add_argument("--include_last_layer", action="store_true",
                       help="Include the last layer even if not in step sequence")
    parser.add_argument("--layer_start", type=int, default=None,
                       help="Start layer index for range mode (inclusive)")
    parser.add_argument("--layer_end", type=int, default=None,
                       help="End layer index for range mode (inclusive)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.layer_start is not None and args.layer_end is None:
        parser.error("--layer_end is required when --layer_start is provided")
    if args.layer_end is not None and args.layer_start is None:
        parser.error("--layer_start is required when --layer_end is provided")
    
    create_timestep_comparison_grid(
        images_dir=args.images_dir,
        example_i=args.example_i,
        kernel_size=args.kernel_size,
        timesteps=args.timesteps,
        grad_type=args.grad_type,
        output_path=args.output_path,
        layer_step=args.layer_step,
        include_last_layer=args.include_last_layer,
        layer_start=args.layer_start,
        layer_end=args.layer_end
    )

if __name__ == "__main__":
    main()


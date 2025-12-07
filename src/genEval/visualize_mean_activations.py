#!/usr/bin/env python3
"""
Script to visualize mean activations from collected ablation data.

For a given component, timestep, and layer, generates:
1. PCA RGB map: Projects the high-dimensional activations to 3D RGB space using PCA
2. Cosine similarity map: Shows cosine similarity of each pixel's latent with the center pixel
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import argparse


def load_mean_activations(mean_ablations_dir, component, device='cuda'):
    """Load mean activations tensor for a given component."""
    file_path = Path(mean_ablations_dir) / f"mean_activations_{component}.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"Mean activations file not found: {file_path}")
    
    data = torch.load(file_path, map_location=device)
    return data


def extract_activations(data, layer, timestep):
    """
    Extract activations for a specific layer and timestep.
    
    Handles different tensor shapes:
    - mix_ffn: [num_layers, num_timesteps, hidden_dim, H, W]
    - self_attn/cross_attn: [num_layers, num_timesteps, num_tokens, hidden_dim]
    
    Args:
        data: Tensor with shape depending on component type
        layer: Layer index (0-indexed)
        timestep: Timestep index (0-indexed)
    
    Returns:
        Tensor of shape [H, W, hidden_dim]
    """
    if layer >= data.shape[0]:
        raise ValueError(f"Layer {layer} out of range. Max layer: {data.shape[0] - 1}")
    if timestep >= data.shape[1]:
        raise ValueError(f"Timestep {timestep} out of range. Max timestep: {data.shape[1] - 1}")
    
    # Extract slice for this layer and timestep
    activations = data[layer, timestep]
    
    # Handle different shapes
    if len(activations.shape) == 3:
        # Shape: [hidden_dim, H, W] (mix_ffn)
        # Transpose to [H, W, hidden_dim]
        activations = activations.permute(1, 2, 0)
    elif len(activations.shape) == 2:
        # Shape: [num_tokens, hidden_dim] (self_attn, cross_attn)
        # num_tokens = 1024 = 32*32, so reshape to [32, 32, hidden_dim]
        num_tokens, hidden_dim = activations.shape
        H = W = int(num_tokens ** 0.5)  # Assuming square spatial layout
        if H * W != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens to square spatial layout")
        activations = activations.reshape(H, W, hidden_dim)
    else:
        raise ValueError(f"Unexpected activation shape: {activations.shape}")
    
    return activations


def generate_pca_rgb_map(activations):
    """
    Generate RGB map by projecting activations to 3D using PCA.
    
    Args:
        activations: Tensor of shape [H, W, hidden_dim] on GPU
    
    Returns:
        RGB image of shape [H, W, 3] with values in [0, 1] as numpy array
    """
    H, W, hidden_dim = activations.shape
    
    # Reshape to [H*W, hidden_dim] and move to CPU for sklearn
    activations_flat = activations.reshape(-1, hidden_dim).cpu().numpy()
    
    # Normalize data before PCA (standardization: z-score normalization)
    # This ensures PCA is not dominated by features with larger scales
    mean = activations_flat.mean(axis=0, keepdims=True)
    std = activations_flat.std(axis=0, keepdims=True)
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    activations_normalized = (activations_flat - mean) / std
    
    # Apply PCA to get 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(activations_normalized)
    
    # Normalize to [0, 1] range for each component for visualization
    for i in range(3):
        comp = pca_result[:, i]
        comp_min = comp.min()
        comp_max = comp.max()
        if comp_max > comp_min:
            pca_result[:, i] = (comp - comp_min) / (comp_max - comp_min)
        else:
            pca_result[:, i] = 0.5  # Constant value
    
    # Reshape back to [H, W, 3]
    rgb_map = pca_result.reshape(H, W, 3)
    
    return rgb_map


def generate_cosine_similarity_map(activations, reference_pixel='center'):
    """
    Generate cosine similarity map with respect to a reference pixel.
    
    Args:
        activations: Tensor of shape [H, W, hidden_dim] on GPU
        reference_pixel: 'center' for center pixel (H//2, W//2) or 'top_left' for (0, 0)
    
    Returns:
        Similarity map of shape [H, W] with values in [-1, 1] as numpy array
    """
    H, W, hidden_dim = activations.shape
    
    # Get reference pixel latent
    if reference_pixel == 'center':
        ref_h = H // 2
        ref_w = W // 2
    elif reference_pixel == 'top_left':
        ref_h = 0
        ref_w = 0
    else:
        raise ValueError(f"Unknown reference_pixel: {reference_pixel}. Use 'center' or 'top_left'")
    
    reference_latent = activations[ref_h, ref_w]  # [hidden_dim]
    
    # Normalize reference latent
    ref_norm = torch.norm(reference_latent)
    if ref_norm == 0:
        # If reference is zero, return zeros
        return torch.zeros(H, W, device=activations.device).cpu().numpy()
    
    reference_normalized = reference_latent / ref_norm
    
    # Reshape activations to [H*W, hidden_dim] for efficient computation
    activations_flat = activations.reshape(-1, hidden_dim)  # [H*W, hidden_dim]
    
    # Normalize all pixels at once
    pixel_norms = torch.norm(activations_flat, dim=1, keepdim=True)  # [H*W, 1]
    # Avoid division by zero
    pixel_norms = torch.clamp(pixel_norms, min=1e-8)
    activations_normalized = activations_flat / pixel_norms  # [H*W, hidden_dim]
    
    # Compute cosine similarity: dot product with reference normalized vector
    # [H*W, hidden_dim] @ [hidden_dim] -> [H*W]
    similarity_flat = torch.matmul(activations_normalized, reference_normalized)
    
    # Reshape back to [H, W]
    similarity_map = similarity_flat.reshape(H, W)
    
    return similarity_map.cpu().numpy()


def visualize_maps(pca_rgb_map, cosine_sim_map, output_path, component, layer, timestep, reference_pixel='center'):
    """
    Create and save visualization of both maps.
    
    Args:
        pca_rgb_map: RGB image of shape [H, W, 3]
        cosine_sim_map: Similarity map of shape [H, W]
        output_path: Path to save the figure
        component: Component name
        layer: Layer index
        timestep: Timestep index
        reference_pixel: 'center' or 'top_left' for the reference pixel used
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot PCA RGB map
    axes[0].imshow(pca_rgb_map)
    axes[0].set_title(f'PCA RGB Map\n{component}, Layer {layer}, Timestep {timestep}', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot cosine similarity map
    ref_label = 'Center' if reference_pixel == 'center' else 'Top-Left (0,0)'
    im = axes[1].imshow(cosine_sim_map, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title(f'Cosine Similarity to {ref_label}\n{component}, Layer {layer}, Timestep {timestep}',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar for cosine similarity
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize mean activations with PCA RGB and cosine similarity maps"
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        choices=["self_attn", "cross_attn", "mix_ffn"],
        help="Component to visualize"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index (0-indexed)"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help="Timestep index (0-indexed)"
    )
    parser.add_argument(
        "--mean_ablations_dir",
        type=str,
        default="mean_ablations",
        help="Directory containing mean activations files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mean_ablations_visualizations",
        help="Directory to save output visualizations"
    )
    parser.add_argument(
        "--save_pca_only",
        action="store_true",
        help="Save only PCA RGB map (not cosine similarity)"
    )
    parser.add_argument(
        "--save_cosine_only",
        action="store_true",
        help="Save only cosine similarity map (not PCA)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computations (default: cuda)"
    )
    parser.add_argument(
        "--reference_pixel",
        type=str,
        default="center",
        choices=["center", "top_left"],
        help="Reference pixel for cosine similarity: 'center' (default) or 'top_left' (0,0)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load mean activations
    print(f"Loading mean activations for component: {args.component}")
    data = load_mean_activations(args.mean_ablations_dir, args.component, device=device)
    print(f"Data shape: {data.shape}")
    
    # Extract activations for specified layer and timestep
    print(f"Extracting activations for layer {args.layer}, timestep {args.timestep}")
    activations = extract_activations(data, args.layer, args.timestep)
    print(f"Activations shape: {activations.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate maps
    if not args.save_cosine_only:
        print("Generating PCA RGB map...")
        pca_rgb_map = generate_pca_rgb_map(activations)
        print(f"PCA RGB map shape: {pca_rgb_map.shape}")
    
    if not args.save_pca_only:
        print(f"Generating cosine similarity map (reference: {args.reference_pixel})...")
        cosine_sim_map = generate_cosine_similarity_map(activations, reference_pixel=args.reference_pixel)
        print(f"Cosine similarity map shape: {cosine_sim_map.shape}")
    
    # Save individual maps if requested
    if args.save_pca_only:
        output_path = output_dir / f"pca_rgb_{args.component}_layer{args.layer}_timestep{args.timestep}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(pca_rgb_map)
        plt.title(f'PCA RGB Map\n{args.component}, Layer {args.layer}, Timestep {args.timestep}',
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA RGB map to: {output_path}")
        plt.close()
    
    elif args.save_cosine_only:
        ref_label = 'Center' if args.reference_pixel == 'center' else 'Top-Left (0,0)'
        output_path = output_dir / f"cosine_sim_{args.component}_layer{args.layer}_timestep{args.timestep}_ref{args.reference_pixel}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(cosine_sim_map, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Cosine Similarity to {ref_label}\n{args.component}, Layer {args.layer}, Timestep {args.timestep}',
                  fontsize=14, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cosine similarity map to: {output_path}")
        plt.close()
    
    else:
        # Save combined visualization
        ref_suffix = f"_ref{args.reference_pixel}" if args.reference_pixel != 'center' else ""
        output_path = output_dir / f"visualization_{args.component}_layer{args.layer}_timestep{args.timestep}{ref_suffix}.png"
        visualize_maps(pca_rgb_map, cosine_sim_map, output_path, 
                      args.component, args.layer, args.timestep, reference_pixel=args.reference_pixel)


if __name__ == "__main__":
    main()


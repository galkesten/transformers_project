#!/usr/bin/env python3
"""
Delete self-attention related images from qualitative results.

This script removes images from mean_over_tokens and mean_per_token
experiments that are related to self_attn, as those experiments
were run on H200 and produced flawed evaluation results.
"""

import os
from pathlib import Path

def delete_self_attn_images(base_dir):
    """Delete all images containing 'self_attn' in their path"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Base directory does not exist: {base_dir}")
        return
    
    deleted_count = 0
    deleted_files = []
    
    # Walk through layer-wise directory
    layer_wise_dir = base_path / "qualitative_results" / "layer-wise"
    
    if not layer_wise_dir.exists():
        print(f"Layer-wise directory does not exist: {layer_wise_dir}")
        return
    
    # Process each layer directory
    for layer_dir in sorted(layer_wise_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        
        print(f"\nProcessing {layer_dir.name}...")
        
        # Process mean_over_tokens and mean_per_token
        for ablation_type in ["mean_over_tokens", "mean_per_token"]:
            ablation_dir = layer_dir / ablation_type
            
            if not ablation_dir.exists():
                continue
            
            print(f"  Checking {ablation_type}...")
            
            # Recursively find all files
            for file_path in ablation_dir.rglob("*"):
                if file_path.is_file():
                    # Check if filename or path contains 'self_attn'
                    if "self_attn" in file_path.name or "self_attn" in str(file_path):
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            deleted_files.append(str(file_path))
                            if deleted_count % 10 == 0:
                                print(f"    Deleted {deleted_count} files...")
                        except Exception as e:
                            print(f"    ERROR deleting {file_path}: {e}")
    
    # Clean up empty directories
    print(f"\nCleaning up empty directories...")
    for layer_dir in sorted(layer_wise_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        
        for ablation_type in ["mean_over_tokens", "mean_per_token"]:
            ablation_dir = layer_dir / ablation_type
            if ablation_dir.exists():
                # Recursively remove empty directories
                try:
                    for dirpath, dirnames, filenames in os.walk(ablation_dir, topdown=False):
                        dir_path = Path(dirpath)
                        if not any(dir_path.iterdir()):  # Directory is empty
                            dir_path.rmdir()
                            print(f"  Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"  Error cleaning up directories: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files deleted: {deleted_count}")
    print(f"{'='*60}")
    
    if deleted_files and len(deleted_files) <= 20:
        print(f"\nDeleted files:")
        for f in deleted_files:
            print(f"  {f}")
    elif deleted_files:
        print(f"\nFirst 10 deleted files:")
        for f in deleted_files[:10]:
            print(f"  {f}")
        print(f"  ... and {len(deleted_files) - 10} more")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Delete self-attention images from qualitative results"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="ablation_results_new",
        help="Base directory containing qualitative_results (default: ablation_results_new)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted")
        print("="*60)
    
    base_path = Path(args.base_dir)
    layer_wise_dir = base_path / "qualitative_results" / "layer-wise"
    
    if not layer_wise_dir.exists():
        print(f"Error: Directory not found: {layer_wise_dir}")
        return
    
    if args.dry_run:
        # Count files that would be deleted
        count = 0
        for layer_dir in sorted(layer_wise_dir.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            for ablation_type in ["mean_over_tokens", "mean_per_token"]:
                ablation_dir = layer_dir / ablation_type
                if ablation_dir.exists():
                    for file_path in ablation_dir.rglob("*"):
                        if file_path.is_file():
                            if "self_attn" in file_path.name or "self_attn" in str(file_path):
                                count += 1
                                if count <= 20:
                                    print(f"Would delete: {file_path}")
        print(f"\nTotal files that would be deleted: {count}")
    else:
        # Confirm before deleting
        print("This will delete all self-attention related images from:")
        print(f"  {layer_wise_dir}")
        print("\nAblation types: mean_over_tokens, mean_per_token")
        print("Component: self_attn")
        print("\nProceed? (yes/no): ", end="")
        
        response = input().strip().lower()
        if response != "yes":
            print("Cancelled.")
            return
        
        delete_self_attn_images(args.base_dir)

if __name__ == "__main__":
    main()


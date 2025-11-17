#!/bin/bash

# Template command for train_probes_online.py
# This script trains linear probes on diffusion model activations online (without saving activations first)

# ============================================================================
# BASIC EXAMPLE - Train probes for mix_ffn component on all layers and timesteps
# ============================================================================
python src/train_probes_online.py \
    --n_train 100 \
    --n_eval 20 \
    --num_epochs 20 \
    --batch_size 2 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical Horizontal \
    --component_types mix_ffn \
    --timesteps all \
    --layers all \
    --output_csv results/basic_probes.csv

# ============================================================================
# MULTIPLE COMPONENTS - Train probes for all three component types
# ============================================================================
python src/train_probes_online.py \
    --n_train 200 \
    --n_eval 50 \
    --num_epochs 30 \
    --batch_size 4 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical Horizontal Gaussian \
    --component_types self_attn cross_attn mix_ffn \
    --timesteps all \
    --layers all \
    --output_csv results/multi_component_probes.csv

# ============================================================================
# SPECIFIC TIMESTEPS AND LAYERS - Train on specific timesteps and layers
# ============================================================================
python src/train_probes_online.py \
    --n_train 150 \
    --n_eval 30 \
    --num_epochs 25 \
    --batch_size 2 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical \
    --component_types mix_ffn \
    --timesteps 0 5 10 15 19 \
    --layers 0 5 10 15 20 \
    --output_csv results/specific_timesteps_layers.csv

# ============================================================================
# CONTRIBUTIONS MODE - Train probes after gate mechanisms
# ============================================================================
python src/train_probes_online.py \
    --n_train 200 \
    --n_eval 50 \
    --num_epochs 30 \
    --batch_size 4 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical Horizontal \
    --component_types self_attn mix_ffn \
    --timesteps all \
    --layers all \
    --use_contributions_mode \
    --output_csv results/contributions_mode_probes.csv

# ============================================================================
# BLOCK OUTPUT - Train one probe on entire block output instead of components
# ============================================================================
python src/train_probes_online.py \
    --n_train 150 \
    --n_eval 30 \
    --num_epochs 25 \
    --batch_size 2 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical \
    --timesteps all \
    --layers all \
    --train_on_block_output \
    --output_csv results/block_output_probes.csv

# ============================================================================
# INITIAL AND FINAL REPRESENTATIONS - Hook to patch_embed and proj_out
# ============================================================================
python src/train_probes_online.py \
    --n_train 200 \
    --n_eval 50 \
    --num_epochs 30 \
    --batch_size 4 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical Horizontal \
    --component_types mix_ffn \
    --timesteps all \
    --layers all \
    --hook_patch_embed \
    --hook_proj_out \
    --output_csv results/initial_final_probes.csv

# ============================================================================
# FULL EXAMPLE - All features enabled with saving probes and images
# ============================================================================
python src/train_probes_online.py \
    --n_train 300 \
    --n_eval 100 \
    --num_epochs 50 \
    --batch_size 4 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical Horizontal Gaussian \
    --component_types self_attn cross_attn mix_ffn \
    --timesteps all \
    --layers all \
    --use_contributions_mode \
    --normalize_latents_with_layer_norm \
    --hook_patch_embed \
    --hook_proj_out \
    --save_probes \
    --save_images \
    --probes_output_dir probes/full_experiment \
    --images_output_dir eval_images/full_experiment \
    --output_csv results/full_experiment.csv

# ============================================================================
# STEP-BASED SELECTION - Use step sizes for layers and timesteps
# ============================================================================
python src/train_probes_online.py \
    --n_train 200 \
    --n_eval 50 \
    --num_epochs 30 \
    --batch_size 4 \
    --seed 42 \
    --kernel_size 1 \
    --gradient_types Vertical \
    --component_types mix_ffn \
    --timesteps all \
    --layers all \
    --timesteps_step 5 \
    --layers_step 3 \
    --output_csv results/stepped_selection.csv

# ============================================================================
# MINIMAL EXAMPLE - Just the required arguments
# ============================================================================
python src/train_probes_online.py \
    --n_train 50 \
    --n_eval 10 \
    --output_csv results/minimal.csv

# ============================================================================
# ARGUMENT REFERENCE:
# ============================================================================
# Required:
#   --n_train <int>              Number of training prompts
#   --n_eval <int>               Number of evaluation prompts
#   --output_csv <str>           Output CSV file path
#
# Optional (with defaults):
#   --num_epochs <int>           Number of training epochs (default: 20)
#   --batch_size <int>           Batch size for prompts (default: 1)
#   --seed <int>                 Random seed (default: 42)
#   --kernel_size <int>          Kernel size for probe (default: 1)
#   --gradient_types <str>...    Gradient types: Vertical, Horizontal, Gaussian (default: ["Vertical"])
#   --component_types <str>...   Component types: self_attn, cross_attn, mix_ffn (default: ["mix_ffn"])
#   --timesteps <str>...         Specific timesteps (e.g., "0 5 10") or "all" (default: [])
#   --layers <str>...            Specific layers (e.g., "0 5 10") or "all" (default: [])
#   --layers_step <int>          Layer step size (default: -1, no stepping)
#   --timesteps_step <int>       Timestep step size (default: -1, no stepping)
#   --probes_output_dir <str>    Directory to save probes (default: "probes")
#   --images_output_dir <str>    Directory to save images (default: "eval_images")
#
# Flags (no arguments):
#   --use_contributions_mode              Use contributions mode (hook after gates)
#   --normalize_latents_with_layer_norm  Normalize latents with layer norm
#   --train_on_block_output               Train on entire block output instead of components
#   --hook_patch_embed                    Hook to patch_embed output (initial representation)
#   --hook_proj_out                       Hook to proj_out input (final representation)
#   --save_probes                         Save probe models as .pt files
#   --save_images                         Save evaluation images


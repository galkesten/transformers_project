import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import json
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import csv
from filelock import FileLock

print(f"PYTHON EXECUTABLE: {sys.executable}", flush=True)
print(f"CONDA ENV: {os.environ.get('CONDA_DEFAULT_ENV')}", flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run complete ablation experiment pipeline")
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples to generate per layer",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    
    parser.add_argument(
        "--ablation_type",
        choices=["zero", "mean_per_token", "mean_over_tokens", "none"],
        default="none",
        help="type of ablation to apply",
    )
    
    parser.add_argument(
        "--ablation_component",
        choices=["self_attn", "cross_attn", "mix_ffn"],
        default="mix_ffn",
        help="component to apply ablation to",
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run baseline experiment (no ablation, single run)",
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="src/genEval/prompts/evaluation_metadata.jsonl",
        help="JSONL file containing prompts for evaluation",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_results",
        help="directory to save final results and sample images",
    )
    
    parser.add_argument(
        "--sample_images_per_layer",
        type=int,
        default=3,
        help="number of sample images to save per layer",
    )

    parser.add_argument(
        "--mean_activations_file",
        type=str,
        default=None,
        help="file to load mean activations from",
    )

    parser.add_argument(
        "--step_wise",
        action="store_true",
        help="apply ablation step by step",
    )
    return parser.parse_args()

def load_model_info():
    """Get model information to determine number of layers"""
    try:
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        n_layers = len(pipe.transformer.transformer_blocks)
        del pipe  # Free memory
        return n_layers
    except Exception as e:
        print(f"Error loading model: {e}")
        return 20  # Default fallback

def run_generation_layer_wise(layer, args, temp_dir, mean_activations_file=None):
    """Run image generation for a specific layer"""
    print(f"\n🖼️  Generating images for layer {layer}...")
    
    cmd = [
        sys.executable, "src/genEval/generation/diffusers_generate.py",
        args.prompts_file,
        "--outdir", temp_dir,
        "--n_samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--ablation_type", args.ablation_type,
        "--ablation_layer", str(layer),
        "--ablation_component", args.ablation_component,
    ]
    
    # Only add mean_activations_file if it's not None
    if mean_activations_file is not None:
        cmd.extend(["--mean_activations_file", mean_activations_file])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Generation failed for layer {layer}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f" Generation completed for layer {layer}")
    return True

def run_generation_step_wise(args, temp_dir, timestep, mean_activations_file=None):
    """Run image generation for a specific timestep"""
    print(f"\n🖼️  Generating images for timestep {timestep}...")
    
    cmd = [
        sys.executable, "src/genEval/generation/diffusers_generate.py",
        args.prompts_file,
        "--outdir", temp_dir, 
        "--n_samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--ablation_type", args.ablation_type,
        "--ablation_component", args.ablation_component,
        "--timesteps", str(timestep),
        "--step_wise",
    ]

    if mean_activations_file is not None:
        cmd.extend(["--mean_activations_file", mean_activations_file])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Generation failed for timestep {timestep}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f" Generation completed for timestep {timestep}")
    return True

def run_generation_baseline(args, temp_dir):
    """Run image generation for baseline (no ablation)"""
    print(f"\n🖼️  Generating baseline images (no ablation)...")
    
    cmd = [
        sys.executable, "src/genEval/generation/diffusers_generate.py",
        args.prompts_file,
        "--outdir", temp_dir,
        "--n_samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--ablation_type", "none",  # No ablation for baseline
        "--ablation_layer", "0",    # Dummy layer
        "--ablation_component", args.ablation_component,
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Baseline generation failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f" Baseline generation completed")
    return True

def run_evaluation(temp_dir, args):
    """Run evaluation on generated images"""
    print(f"\n🔍 Evaluating images in {temp_dir}...")
    
    results_file = os.path.join(temp_dir, "evaluation_results.jsonl")
    
    cmd = [
        sys.executable, "src/genEval/evaluation/evaluate_images.py",
        temp_dir,
        "--outfile", results_file,
         "--model-path",  "src/genEval/object_detector_folder"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f" Evaluation failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f" Evaluation completed")
    return results_file

def extract_results_to_csv(results_file, layer_or_timestep, args, output_dir, all_results_csv):
    
    """Extract evaluation results and append summary row to CSV with file locking"""
    print(f"\n Extracting results for layer {layer_or_timestep}...")
    
    try:
        # Load results
        df = pd.read_json(results_file, orient="records", lines=True)
        
        # Calculate summary statistics (ONE ROW PER LAYER)
        if args.baseline:
            summary = {
                'experiment_type': 'baseline',
                'ablation_type': 'none',
                'ablation_component': args.ablation_component,
                'total_images': len(df),
                'total_prompts': len(df.groupby('metadata')),
                'correct_images_pct': df['correct'].mean(),
                'correct_prompts_pct': df.groupby('metadata')['correct'].any().mean(),
            }
        elif args.step_wise:
            summary = {
                'timestep': layer_or_timestep,
                'ablation_type': args.ablation_type,
                'ablation_component': args.ablation_component,
                'total_images': len(df),
                'total_prompts': len(df.groupby('metadata')),
                'correct_images_pct': df['correct'].mean(),
                'correct_prompts_pct': df.groupby('metadata')['correct'].any().mean(),
            }
        else:
            summary = {
                'layer': layer_or_timestep,
                'ablation_type': args.ablation_type,
                'ablation_component': args.ablation_component,
                'total_images': len(df),
                'total_prompts': len(df.groupby('metadata')),
                'correct_images_pct': df['correct'].mean(),
                'correct_prompts_pct': df.groupby('metadata')['correct'].any().mean(),
            }
        
        # Calculate task breakdown
        task_scores = []
        for tag, task_df in df.groupby('tag', sort=False):
            task_score = task_df['correct'].mean()
            task_scores.append(task_score)
            summary[f'task_{tag}_score'] = task_score
        
        summary['overall_score'] = pd.Series(task_scores).mean()
        
        # Write to CSV with file locking using FileLock
        os.makedirs(os.path.dirname(all_results_csv), exist_ok=True)
        lock_path = all_results_csv + ".lock"
        lock = FileLock(lock_path)
        
        with lock:
            write_header = not os.path.exists(all_results_csv)
            with open(all_results_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    # Write header
                    header = list(summary.keys())
                    writer.writerow(header)
                # Write data row
                row = list(summary.values())
                writer.writerow(row)
        
        print(f"Summary results appended to {all_results_csv}")
        return summary
                    
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None



def save_sample_images(temp_dir, layer_or_timestep, args, output_dir):
    """Save sample images from the generated images (including subfolders)"""
    if args.baseline:
        print(f"\n  Saving baseline sample images...")
    elif args.step_wise:
        print(f"\n  Saving sample images for timestep {layer_or_timestep}...")
    else:
        print(f"\n  Saving sample images for layer {layer_or_timestep}...")

    try:
        # Find image files in temp_dir and all subfolders
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        temp_path = Path(temp_dir)
        for ext in image_extensions:
            image_files.extend(temp_path.rglob(f"*{ext}"))

        if not image_files:
            print(f"No image files found in {temp_dir} or its subfolders")
            return

        if args.baseline:
            # Create baseline directory
            sample_dir = os.path.join(output_dir, "sample_images", "baseline")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Copy sample images
            sample_count = min(args.sample_images_per_layer, len(image_files))
            selected_files = image_files[:sample_count]

            for i, img_file in enumerate(selected_files):
                new_name = f"baseline_sample_{i}{img_file.suffix}"
                dest_path = os.path.join(sample_dir, new_name)
                shutil.copy2(img_file, dest_path)

            print(f"Saved {sample_count} baseline sample images to {sample_dir}")
        elif args.step_wise:
            # Create timestep-specific directory
            timestep_dir = os.path.join(output_dir, "sample_images", f"timestep_{layer_or_timestep}")
            os.makedirs(timestep_dir, exist_ok=True)

            # Copy sample images
            sample_count = min(args.sample_images_per_layer, len(image_files))
            selected_files = image_files[:sample_count]

            for i, img_file in enumerate(selected_files):
                new_name = f"timestep_{layer_or_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{i}{img_file.suffix}"
                dest_path = os.path.join(timestep_dir, new_name)
                shutil.copy2(img_file, dest_path)

            print(f"Saved {sample_count} sample images to {timestep_dir}")  
        else:
            # Create layer-specific directory
            layer_dir = os.path.join(output_dir, "sample_images", f"layer_{layer_or_timestep}")
            os.makedirs(layer_dir, exist_ok=True)

            # Copy sample images
            sample_count = min(args.sample_images_per_layer, len(image_files))
            selected_files = image_files[:sample_count]

            for i, img_file in enumerate(selected_files):
                new_name = f"layer_{layer_or_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{i}{img_file.suffix}"
                dest_path = os.path.join(layer_dir, new_name)
                shutil.copy2(img_file, dest_path)

            print(f"Saved {sample_count} sample images to {layer_dir}")

    except Exception as e:
        print(f" Error saving sample images: {e}")




def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "sample_images"), exist_ok=True)
    #load mean activations
    if args.baseline:
        # Baseline mode: run single experiment without ablation
        print(f"Running baseline experiment (no ablation)")
        print(f"Output directory: {args.output_dir}")
        
        # Create the main CSV file path for baseline results
        all_results_csv = os.path.join(args.output_dir, "baseline_results.csv")
        
        # Create temporary directory for baseline
        temp_dir_name = "baseline_experiment"
        with tempfile.TemporaryDirectory(prefix=temp_dir_name + "_") as temp_dir:
            
            # Step 1: Generate images (no ablation)
            if not run_generation_baseline(args, temp_dir):
                print(f"Baseline generation failed")
                return

            # Step 2: Evaluate images
            results_file = run_evaluation(temp_dir, args)
            if results_file is None:
                print(f"Baseline evaluation failed")
                return
            
            # Step 3: Extract results to CSV
            extract_results_to_csv(results_file, None, args, args.output_dir, all_results_csv)
            
            # Step 4: Save sample images
            save_sample_images(temp_dir, None, args, args.output_dir)
            
            # Step 5: Cleanup (handled by context manager)
            print(f" Baseline experiment completed")
        
        print(f"\n Baseline experiment completed!")
        print(f" Results saved in: {args.output_dir}")
        print(f" Main results file: {all_results_csv}")

    elif args.step_wise:
        # Step wise mode: run experiments for all timesteps
        print(f"Running step wise experiment")
        print(f"Output directory: {args.output_dir}")
        print(f"Ablation type: {args.ablation_type}")
        print(f"Ablation component: {args.ablation_component}")

        # Create the main CSV file path for all results
        all_results_csv = os.path.join(args.output_dir, f"all_timesteps_results_{args.ablation_type}_{args.ablation_component}.csv")

        # Run experiment for each timestep
        for timestep in tqdm(range(20), desc="Processing timesteps"):
            print(f"\n{'='*60}")
            print(f"Processing Timestep {timestep}")
            print(f"{'='*60}")

            # Create temporary directory for this timestep
            temp_dir_name = f"ablation_{args.ablation_type}_{args.ablation_component}_timestep_{timestep}"
            with tempfile.TemporaryDirectory(prefix=temp_dir_name + "_") as temp_dir:
                
                # Step 1: Generate images
                if not run_generation_step_wise(args, temp_dir, timestep, args.mean_activations_file):
                    print(f"Skipping timestep {timestep} due to generation failure")
                    continue

                # Step 2: Evaluate images
                results_file = run_evaluation(temp_dir, args)
                if results_file is None:
                    print(f"Skipping timestep {timestep} due to evaluation failure")
                    continue
                
                # Step 3: Extract results to CSV
                extract_results_to_csv(results_file, timestep, args, args.output_dir, all_results_csv)

                # Step 4: Save sample images
                save_sample_images(temp_dir, timestep, args, args.output_dir)
                
                # Step 5: Cleanup (handled by context manager)
                print(f" Timestep {timestep} processing completed")
    else:
        # Ablation mode: run experiments for all layers
        # Get number of layers
        n_layers = load_model_info()
        print(f"Running ablation experiment on {n_layers} layers")
        print(f"Ablation type: {args.ablation_type}")
        print(f"Ablation component: {args.ablation_component}")
        print(f"Output directory: {args.output_dir}")
        
        # Create the main CSV file path for all results
        all_results_csv = os.path.join(args.output_dir, f"all_layers_results_{args.ablation_type}_{args.ablation_component}.csv")
        
        # Run experiment for each layer
        for layer in tqdm(range(n_layers), desc="Processing layers"):
            print(f"\n{'='*60}")
            print(f"Processing Layer {layer}/{n_layers-1}")
            print(f"{'='*60}")
            
            # Create temporary directory for this layer
            temp_dir_name = f"ablation_{args.ablation_type}_{args.ablation_component}_layer_{layer}"
            with tempfile.TemporaryDirectory(prefix=temp_dir_name + "_") as temp_dir:
                
                # Step 1: Generate images
                if not run_generation_layer_wise(layer, args, temp_dir, args.mean_activations_file):
                    print(f"Skipping layer {layer} due to generation failure")
                    continue

                # Step 2: Evaluate images
                results_file = run_evaluation(temp_dir, args)
                if results_file is None:
                    print(f"Skipping layer {layer} due to evaluation failure")
                    continue
                
                # Step 3: Extract results to CSV
                extract_results_to_csv(results_file, layer, args, args.output_dir, all_results_csv)
                
                # Step 4: Save sample images
                save_sample_images(temp_dir, layer, args, args.output_dir)
                
                # Step 5: Cleanup (handled by context manager)
                print(f" Layer {layer} processing completed")
        
        print(f"\n Ablation experiment completed!")
        print(f" Results saved in: {args.output_dir}")
        print(f" Main results file: {all_results_csv}")

if __name__ == "__main__":
    main() 
    
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
import re
from collections import defaultdict

print(f"PYTHON EXECUTABLE: {sys.executable}", flush=True)
print(f"CONDA ENV: {os.environ.get('CONDA_DEFAULT_ENV')}", flush=True)

def sanitize_prompt_name(prompt):
    """Convert prompt to valid folder name"""
    # Remove special characters, replace spaces with underscores
    name = re.sub(r'[^\w\s-]', '', prompt.lower())
    name = re.sub(r'[-\s]+', '_', name)
    return name.strip('_')

def get_sampled_prompts(prompts_file, n_per_task):
    """Get sampled prompts (deterministic, no file I/O for thread safety)"""
    with open(prompts_file) as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]
    
    # Group by task
    tasks = defaultdict(list)
    for prompt_data in all_prompts:
        task = prompt_data.get('tag', 'unknown')
        tasks[task].append(prompt_data)
    
    # Sample N per task (deterministic: first N)
    sampled_prompt_texts = set()
    for task, prompts in tasks.items():
        n_sample = min(n_per_task, len(prompts))
        for prompt_data in prompts[:n_sample]:
            sampled_prompt_texts.add(prompt_data['prompt'])
    
    return sampled_prompt_texts

def get_prompt_success_status(temp_dir, sampled_prompts, n_succeeded_per_task, n_failed_per_task):
    """Categorize prompts by success/failure based on evaluation results
    
    Args:
        temp_dir: Directory with evaluation_results.jsonl
        sampled_prompts: Optional - Set of prompts to filter (None = use all)
        n_succeeded_per_task: Number of succeeded prompts to select per task
        n_failed_per_task: Number of failed prompts to select per task
    
    Returns dict: {prompt_text: 'succeeded' or 'failed'}
    """
    evaluation_file = os.path.join(temp_dir, "evaluation_results.jsonl")
    
    if not os.path.exists(evaluation_file):
        return {}  # No evaluation results yet
    
    # Load evaluation results
    df = pd.read_json(evaluation_file, orient="records", lines=True)
    
    # Optionally filter to specific prompts
    if sampled_prompts is not None:
        df = df[df['prompt'].isin(sampled_prompts)]
    
    # Group by task and prompt, check if any image succeeded
    # A prompt is "succeeded" if at least one image is correct
    prompt_success = df.groupby(['tag', 'prompt'])['correct'].any().reset_index()
    prompt_success.columns = ['task', 'prompt', 'succeeded']
    
    # Select N succeeded and N failed per task (from the sampled set)
    selected_prompts = {}
    
    for task in prompt_success['task'].unique():
        task_df = prompt_success[prompt_success['task'] == task]
        
        # Get succeeded and failed prompts
        succeeded_prompts = task_df[task_df['succeeded']]['prompt'].tolist()
        failed_prompts = task_df[~task_df['succeeded']]['prompt'].tolist()
        
        # Take first N of each (deterministic)
        for prompt in succeeded_prompts[:n_succeeded_per_task]:
            selected_prompts[prompt] = 'succeeded'
        
        for prompt in failed_prompts[:n_failed_per_task]:
            selected_prompts[prompt] = 'failed'
    
    return selected_prompts

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
        "--sample_prompts_per_task",
        type=int,
        default=5,
        help="number of prompts to sample per task type for qualitative analysis",
    )
    
    parser.add_argument(
        "--sample_succeeded_per_task",
        type=int,
        default=3,
        help="number of succeeded prompts to save per task",
    )
    
    parser.add_argument(
        "--sample_failed_per_task",
        type=int,
        default=3,
        help="number of failed prompts to save per task",
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
    
    parser.add_argument(
        "--baseline_results_file",
        type=str,
        default=None,
        help="path to baseline evaluation results for comparison (optional)",
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
        args.prompts_file,  # Use ALL prompts for evaluation
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

def load_timestep_mapping(temp_dir):
    """Load timestep mapping from generation output"""
    mapping_file = os.path.join(temp_dir, "timestep_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            return {int(k): v for k, v in mapping.items()}
    return None

def get_quantitative_results_dir(output_dir, ablation_type, ablation_component, step_wise=False):
    """Get the directory for quantitative results based on ablation type and component"""
    if ablation_type == "none":
        # Baseline goes directly in quantitative_results
        return os.path.join(output_dir, "quantitative_results", "baseline")
    else:
        # Separate layer-wise and step-wise results
        experiment_type = "step-wise" if step_wise else "layer-wise"
        return os.path.join(output_dir, "quantitative_results", experiment_type, ablation_type, ablation_component)

def save_detailed_results(results_file, layer_or_timestep, args, output_dir, actual_timestep=None):
    """Save detailed success/failure information for each prompt"""
    try:
        df = pd.read_json(results_file, orient="records", lines=True)
        
        # Create results directory organized by ablation type and component
        results_dir = get_quantitative_results_dir(output_dir, args.ablation_type, args.ablation_component, args.step_wise)
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine filename based on experiment type
        if args.baseline:
            detailed_file = os.path.join(results_dir, "baseline_detailed_results.jsonl")
        elif args.step_wise:
            detailed_file = os.path.join(results_dir, 
                f"iteration_{layer_or_timestep}_timestep_{actual_timestep}_detailed_results.jsonl")
        else:
            detailed_file = os.path.join(results_dir, 
                f"layer_{layer_or_timestep}_detailed_results.jsonl")
        
        # Save the detailed results
        df.to_json(detailed_file, orient="records", lines=True)
        print(f"Saved detailed results to {detailed_file}")
        
    except Exception as e:
        print(f"Error saving detailed results: {e}")

def extract_results_to_csv(results_file, layer_or_timestep, args, output_dir, actual_timestep=None):
    
    """Extract evaluation results and append summary row to CSV with file locking"""
    print(f"\n Extracting results for layer {layer_or_timestep}...")
    
    try:
        # Load results
        df = pd.read_json(results_file, orient="records", lines=True)
        
        # Get organized results directory
        results_dir = get_quantitative_results_dir(output_dir, args.ablation_type, args.ablation_component, args.step_wise)
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine CSV filename based on experiment type
        if args.baseline:
            csv_filename = "baseline_results.csv"
        elif args.step_wise:
            csv_filename = f"all_timesteps_results_{args.ablation_type}_{args.ablation_component}.csv"
        else:
            csv_filename = f"all_layers_results_{args.ablation_type}_{args.ablation_component}.csv"
        
        all_results_csv = os.path.join(results_dir, csv_filename)
        
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
                'iteration_index': layer_or_timestep,
                'actual_timestep': actual_timestep if actual_timestep is not None else layer_or_timestep,
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



def save_sample_images(temp_dir, layer_or_timestep, args, output_dir, actual_timestep=None):
    """Save sampled images (first N per task, no categorization)"""
    if args.baseline:
        print(f"\n  Saving baseline images (all prompts)...")
    elif args.step_wise:
        print(f"\n  Saving sampled images for timestep {actual_timestep}...")
    else:
        print(f"\n  Saving sampled images for layer {layer_or_timestep}...")

    try:
        # Simple sampling: first N prompts per task
        if args.baseline:
            sampled_prompt_texts = None  # Save all
        else:
            sampled_prompt_texts = get_sampled_prompts(args.prompts_file, args.sample_prompts_per_task)
        
        temp_path = Path(temp_dir)
        
        # Iterate through each prompt folder (00000, 00001, etc.)
        prompt_folders = sorted([d for d in temp_path.iterdir() if d.is_dir()])
        
        saved_count = 0
        for prompt_folder in prompt_folders:
            # Load metadata to get prompt info
            metadata_file = prompt_folder / "metadata.jsonl"
            if not metadata_file.exists():
                continue
                
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            prompt_text = metadata['prompt']
            
            # Skip if not in sampled prompts (except for baseline which saves all)
            if sampled_prompt_texts is not None and prompt_text not in sampled_prompt_texts:
                continue
            
            task = metadata.get('tag', 'unknown')
            prompt_name = sanitize_prompt_name(prompt_text)
            saved_count += 1
            
            # Find all sample images (exclude grid.png!)
            samples_dir = prompt_folder / "samples"
            if not samples_dir.exists():
                continue
            
            image_files = sorted(samples_dir.glob("*.png"))
            
            if not image_files:
                continue
            
            # Determine target directory structure (NO succeeded/failed subfolder)
            if args.baseline:
                target_base = os.path.join(output_dir, "qualitative_results", "baseline", task, prompt_name)
            elif args.step_wise:
                target_base = os.path.join(output_dir, "qualitative_results", "step-wise", 
                                          f"timestep_{actual_timestep}", args.ablation_type, task, "sampled", prompt_name)
            else:
                target_base = os.path.join(output_dir, "qualitative_results", "layer-wise",
                                          f"layer_{layer_or_timestep}", args.ablation_type, task, "sampled", prompt_name)
            
            os.makedirs(target_base, exist_ok=True)
            
            # Copy all sample images with descriptive names
            for sample_idx, img_file in enumerate(image_files):
                if args.baseline:
                    new_name = f"{prompt_name}_baseline_sample_{sample_idx}.png"
                elif args.step_wise:
                    new_name = f"{prompt_name}_timestep_{actual_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                else:
                    new_name = f"{prompt_name}_layer_{layer_or_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                
                dest_path = os.path.join(target_base, new_name)
                shutil.copy2(img_file, dest_path)
        
        if args.baseline:
            print(f"  Saved images for all {saved_count} prompts to qualitative_results/")
        else:
            print(f"  Saved {saved_count} sampled prompts to qualitative_results/")

    except Exception as e:
        print(f" Error saving sample images: {e}")

def save_succeeded_failed_images(temp_dir, layer_or_timestep, args, output_dir, actual_timestep=None):
    """Save images organized by success/failure (N succeeded + N failed per task)"""
    if args.baseline:
        return  # Skip for baseline
    
    if args.step_wise:
        print(f"\n  Saving succeeded/failed images for timestep {actual_timestep}...")
    else:
        print(f"\n  Saving succeeded/failed images for layer {layer_or_timestep}...")

    try:
        # Get prompts categorized by success/failure (independent from sampling)
        prompt_status = get_prompt_success_status(temp_dir, None, 
                                                 args.sample_succeeded_per_task, args.sample_failed_per_task)
        
        if not prompt_status:
            print(f"  No evaluation results available yet")
            return
        
        temp_path = Path(temp_dir)
        prompt_folders = sorted([d for d in temp_path.iterdir() if d.is_dir()])
        
        saved_count = 0
        for prompt_folder in prompt_folders:
            metadata_file = prompt_folder / "metadata.jsonl"
            if not metadata_file.exists():
                continue
                
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            prompt_text = metadata['prompt']
            
            # Skip if not in succeeded/failed categorization
            if prompt_text not in prompt_status:
                continue
            
            task = metadata.get('tag', 'unknown')
            prompt_name = sanitize_prompt_name(prompt_text)
            status = prompt_status[prompt_text]  # 'succeeded' or 'failed'
            saved_count += 1
            
            # Find all sample images (exclude grid.png!)
            samples_dir = prompt_folder / "samples"
            if not samples_dir.exists():
                continue
            
            image_files = sorted(samples_dir.glob("*.png"))
            
            if not image_files:
                continue
            
            # Directory structure with succeeded/failed categorization
            if args.step_wise:
                target_base = os.path.join(output_dir, "qualitative_results", "step-wise", 
                                          f"timestep_{actual_timestep}", args.ablation_type, task, status, prompt_name)
            else:
                target_base = os.path.join(output_dir, "qualitative_results", "layer-wise",
                                          f"layer_{layer_or_timestep}", args.ablation_type, task, status, prompt_name)
            
            os.makedirs(target_base, exist_ok=True)
            
            # Copy all sample images
            for sample_idx, img_file in enumerate(image_files):
                if args.step_wise:
                    new_name = f"{prompt_name}_timestep_{actual_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                else:
                    new_name = f"{prompt_name}_layer_{layer_or_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                
                dest_path = os.path.join(target_base, new_name)
                shutil.copy2(img_file, dest_path)
        
        print(f"  Saved {saved_count} succeeded/failed prompts to qualitative_results/")

    except Exception as e:
        print(f" Error saving succeeded/failed images: {e}")

def save_baseline_comparison_images(temp_dir, layer_or_timestep, args, output_dir, baseline_results_file, actual_timestep=None):
    """Save images where baseline and ablation results differ
    
    Saves two categories:
    1. Succeeded in baseline, failed in ablation
    2. Failed in baseline, succeeded in ablation
    """
    if args.baseline:
        print(f"  Skipping baseline comparison (this IS baseline)", flush=True)
        return
    
    if not baseline_results_file:
        print(f"  Skipping baseline comparison (no baseline file provided)", flush=True)
        return
    
    if args.step_wise:
        print(f"\n  Saving baseline comparison images for timestep {actual_timestep}...", flush=True)
    else:
        print(f"\n  Saving baseline comparison images for layer {layer_or_timestep}...", flush=True)

    try:
        # Load baseline results
        print(f"  Looking for baseline file: {baseline_results_file}", flush=True)
        if not os.path.exists(baseline_results_file):
            print(f"  ERROR: Baseline results file not found: {baseline_results_file}", flush=True)
            return
        
        baseline_df = pd.read_json(baseline_results_file, orient="records", lines=True)
        print(f"  Loaded {len(baseline_df)} baseline results", flush=True)
        
        # Load current ablation results
        ablation_file = os.path.join(temp_dir, "evaluation_results.jsonl")
        print(f"  Looking for ablation file: {ablation_file}", flush=True)
        if not os.path.exists(ablation_file):
            print(f"  ERROR: Ablation results not available yet: {ablation_file}", flush=True)
            return
        
        ablation_df = pd.read_json(ablation_file, orient="records", lines=True)
        print(f"  Loaded {len(ablation_df)} ablation results", flush=True)
        
        # Merge on prompt and sample index to compare same images
        # Extract sample index from filename (format: "00000/samples/00000.png")
        # Handle both 'filename' and 'image_path' column names
        baseline_path_col = 'filename' if 'filename' in baseline_df.columns else 'image_path'
        ablation_path_col = 'filename' if 'filename' in ablation_df.columns else 'image_path'
        
        baseline_df['sample_idx'] = baseline_df[baseline_path_col].str.extract(r'/samples/(\d+)\.png')[0].astype(str)
        ablation_df['sample_idx'] = ablation_df[ablation_path_col].str.extract(r'/samples/(\d+)\.png')[0].astype(str)
        
        print(f"  Extracted sample indices from paths", flush=True)
        print(f"  Baseline sample_idx range: {baseline_df['sample_idx'].min()} to {baseline_df['sample_idx'].max()}", flush=True)
        print(f"  Ablation sample_idx range: {ablation_df['sample_idx'].min()} to {ablation_df['sample_idx'].max()}", flush=True)
        
        # Merge to compare
        comparison = baseline_df.merge(
            ablation_df, 
            on=['prompt', 'tag', 'sample_idx'],
            suffixes=('_baseline', '_ablation'),
            how='inner'
        )
        
        print(f"  Found {len(comparison)} matching images to compare", flush=True)
        
        if len(comparison) == 0:
            print(f"  WARNING: No matching images found! Check if prompts/sample indices match.", flush=True)
            print(f"  Baseline unique prompts: {baseline_df['prompt'].nunique()}", flush=True)
            print(f"  Ablation unique prompts: {ablation_df['prompt'].nunique()}", flush=True)
            return
        
        # Find discrepancies
        # Category 1: Baseline succeeded, ablation failed
        baseline_success_ablation_fail = comparison[
            (comparison['correct_baseline'] == True) & 
            (comparison['correct_ablation'] == False)
        ]
        
        # Category 2: Baseline failed, ablation succeeded
        baseline_fail_ablation_success = comparison[
            (comparison['correct_baseline'] == False) & 
            (comparison['correct_ablation'] == True)
        ]
        
        # Save images for each category
        temp_path = Path(temp_dir)
        
        categories = {
            'baseline_succeeded_ablation_failed': baseline_success_ablation_fail,
            'baseline_failed_ablation_succeeded': baseline_fail_ablation_success
        }
        
        total_saved = 0
        category_counts = {}
        
        for category_name, category_df in categories.items():
            if len(category_df) == 0:
                print(f"    No examples found for {category_name}", flush=True)
                category_counts[category_name] = 0
                continue
            
            print(f"    Processing {category_name}: {len(category_df)} examples", flush=True)
            
            category_saved = 0
            
            # Group by task and take a few per task
            for task in category_df['tag'].unique():
                task_df = category_df[category_df['tag'] == task]
                
                # Take up to 3 prompts per task (or all if fewer)
                unique_prompts = task_df['prompt'].unique()
                n_prompts = min(3, len(unique_prompts))
                
                if n_prompts == 0:
                    continue
                
                # Select first N prompts (deterministic)
                selected_prompts = unique_prompts[:n_prompts]
                
                for prompt_text in selected_prompts:
                    prompt_df = task_df[task_df['prompt'] == prompt_text]
                    prompt_name = sanitize_prompt_name(prompt_text)
                    
                    # Get which samples differ for this prompt
                    differing_samples = sorted([int(idx) for idx in prompt_df['sample_idx'].unique()])
                    
                    # Find the prompt folder
                    for prompt_folder in temp_path.iterdir():
                        if not prompt_folder.is_dir():
                            continue
                        
                        metadata_file = prompt_folder / "metadata.jsonl"
                        if not metadata_file.exists():
                            continue
                        
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        if metadata['prompt'] != prompt_text:
                            continue
                        
                        # Found the right folder
                        samples_dir = prompt_folder / "samples"
                        if not samples_dir.exists():
                            continue
                        
                        # Determine target directory
                        if args.step_wise:
                            target_base = os.path.join(output_dir, "qualitative_results", "step-wise", 
                                                      f"timestep_{actual_timestep}", args.ablation_type, 
                                                      task, category_name, prompt_name)
                        else:
                            target_base = os.path.join(output_dir, "qualitative_results", "layer-wise",
                                                      f"layer_{layer_or_timestep}", args.ablation_type, 
                                                      task, category_name, prompt_name)
                        
                        os.makedirs(target_base, exist_ok=True)
                        
                        # Copy the specific images that differ
                        for _, row in prompt_df.iterrows():
                            sample_idx = int(row['sample_idx'])
                            image_file = samples_dir / f"{sample_idx:05d}.png"
                            
                            if image_file.exists():
                                if args.step_wise:
                                    new_name = f"{prompt_name}_timestep_{actual_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                                else:
                                    new_name = f"{prompt_name}_layer_{layer_or_timestep}_{args.ablation_type}_{args.ablation_component}_sample_{sample_idx}.png"
                                
                                dest_path = os.path.join(target_base, new_name)
                                shutil.copy2(image_file, dest_path)
                                total_saved += 1
                                category_saved += 1
                        
                        print(f"      Saved {len(differing_samples)} differing samples for prompt: {prompt_text[:50]}... (samples: {differing_samples})", flush=True)
                        break  # Found and processed this prompt
            
            category_counts[category_name] = category_saved
        
        # Print summary
        print(f"  Saved {total_saved} baseline comparison images:", flush=True)
        for category_name, count in category_counts.items():
            print(f"    - {category_name}: {count} images", flush=True)

    except Exception as e:
        import traceback
        print(f" ERROR saving baseline comparison images: {e}", flush=True)
        print(f" Traceback:", flush=True)
        traceback.print_exc()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.baseline:
        # Baseline mode: run single experiment without ablation
        print(f"Running baseline experiment (no ablation)")
        print(f"Output directory: {args.output_dir}")
        
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
            
            # Step 3: Save detailed results
            save_detailed_results(results_file, None, args, args.output_dir)
            
            # Step 4: Extract results to CSV
            extract_results_to_csv(results_file, None, args, args.output_dir)
            
            # Step 5: Save sampled images
            save_sample_images(temp_dir, None, args, args.output_dir)
            
            # Step 6: Save succeeded/failed images (separate from sampling)
            save_succeeded_failed_images(temp_dir, None, args, args.output_dir)
            
            # Step 7: Save baseline comparison images (if baseline file provided)
            if args.baseline_results_file:
                save_baseline_comparison_images(temp_dir, None, args, args.output_dir, args.baseline_results_file)
            
            # Step 8: Cleanup (handled by context manager)
            print(f" Baseline experiment completed")
        
        print(f"\n Baseline experiment completed!")
        print(f" Results saved in: {args.output_dir}/quantitative_results/baseline/")

    elif args.step_wise:
        # Step wise mode: run experiments for all timesteps
        print(f"Running step wise experiment")
        print(f"Output directory: {args.output_dir}")
        print(f"Ablation type: {args.ablation_type}")
        print(f"Ablation component: {args.ablation_component}")

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
                
                # Step 2.5: Load timestep mapping
                timestep_mapping = load_timestep_mapping(temp_dir)
                actual_timestep = timestep_mapping.get(timestep) if timestep_mapping else None
                
                # Step 3: Save detailed results
                save_detailed_results(results_file, timestep, args, args.output_dir, actual_timestep=actual_timestep)
                
                # Step 4: Extract results to CSV
                extract_results_to_csv(results_file, timestep, args, args.output_dir, actual_timestep=actual_timestep)

                # Step 5: Save sampled images
                save_sample_images(temp_dir, timestep, args, args.output_dir, actual_timestep=actual_timestep)
                
                # Step 6: Save succeeded/failed images (separate)
                save_succeeded_failed_images(temp_dir, timestep, args, args.output_dir, actual_timestep=actual_timestep)
                
                # Step 7: Save baseline comparison images (if baseline file provided)
                if args.baseline_results_file:
                    save_baseline_comparison_images(temp_dir, timestep, args, args.output_dir, args.baseline_results_file, actual_timestep=actual_timestep)
                
                # Step 8: Cleanup (handled by context manager)
                print(f" Timestep {timestep} processing completed")
    else:
        # Ablation mode: run experiments for all layers
        # Get number of layers
        n_layers = load_model_info()
        print(f"Running ablation experiment on {n_layers} layers")
        print(f"Ablation type: {args.ablation_type}")
        print(f"Ablation component: {args.ablation_component}")
        print(f"Output directory: {args.output_dir}")
        
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
                
                # Step 3: Save detailed results
                save_detailed_results(results_file, layer, args, args.output_dir)
                
                # Step 4: Extract results to CSV
                extract_results_to_csv(results_file, layer, args, args.output_dir)
                
                # Step 5: Save sampled images
                save_sample_images(temp_dir, layer, args, args.output_dir)
                
                # Step 6: Save succeeded/failed images (separate)
                save_succeeded_failed_images(temp_dir, layer, args, args.output_dir)
                
                # Step 7: Save baseline comparison images (if baseline file provided)
                if args.baseline_results_file:
                    save_baseline_comparison_images(temp_dir, layer, args, args.output_dir, args.baseline_results_file)
                
                # Step 8: Cleanup (handled by context manager)
                print(f" Layer {layer} processing completed")
        
        print(f"\n Ablation experiment completed!")
        results_dir = get_quantitative_results_dir(args.output_dir, args.ablation_type, args.ablation_component, args.step_wise)
        print(f" Results saved in: {results_dir}")

if __name__ == "__main__":
    main() 
    
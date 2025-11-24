#!/usr/bin/env python3
"""
Detect problematic losses in probing training loss statistics.

Simple approach:
1. Check if loss is converging (going down over epochs)
2. Check if variance is too high (unstable training)
3. Check for NaN values
"""

import pandas as pd
import numpy as np
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import random


def load_all_loss_stats(results_dir="probing/results_csvs", filename_pattern=None):
    """Load and concatenate loss stats CSV files matching the filename pattern"""
    csv_files = glob.glob(f"{results_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    # Filter by filename pattern if provided
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
        print(f"Filtered to {len(csv_files)} files containing '{filename_pattern}' in filename")
    
    # Only keep loss_stats files
    csv_files = [f for f in csv_files if "loss_stats" in f]
    print(f"Filtered to {len(csv_files)} loss_stats files")
    
    if len(csv_files) == 0:
        raise ValueError(f"No loss_stats CSV files found with pattern '{filename_pattern}'")
    
    # Load and merge all matching CSVs
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merging: {len(combined)}")
    
    return combined


def get_epoch_columns(df):
    """Extract epoch column names from dataframe"""
    all_columns = df.columns.tolist()
    
    # Get unique epoch numbers
    epoch_nums = set()
    for col in all_columns:
        if col.startswith('epoch_') and '_mean_loss' in col:
            epoch_num = int(col.split('_')[1])
            epoch_nums.add(epoch_num)
    
    epoch_nums = sorted(epoch_nums)
    print(f"Found {len(epoch_nums)} epochs: {epoch_nums}")
    return epoch_nums


def detect_nan_losses(df, epoch_nums):
    """Detect rows with NaN values in loss columns"""
    problems = []
    
    for idx, row in df.iterrows():
        has_nan = False
        nan_epochs = []
        
        for epoch in epoch_nums:
            mean_col = f'epoch_{epoch}_mean_loss'
            var_col = f'epoch_{epoch}_var_loss'
            max_col = f'epoch_{epoch}_max_loss'
            
            if pd.isna(row[mean_col]) or pd.isna(row[var_col]) or pd.isna(row[max_col]):
                has_nan = True
                nan_epochs.append(epoch)
        
        if has_nan:
            problems.append({
                'timestep': row['timestep'],
                'position': row['position'],
                'component': row['component'],
                'gradient_type': row['gradient_type'],
                'issue': 'NaN values',
                'details': f'NaN in epochs: {nan_epochs}',
                'severity': 'CRITICAL'
            })
    
    return problems


def detect_non_converging_losses(df, epoch_nums, convergence_window=5):
    """Detect losses that are not converging (not going down)
    
    Strategy: Compare mean loss in early epochs vs late epochs.
    If the loss in the last few epochs is not significantly lower than the first few,
    or if it's increasing, flag it.
    
    Args:
        convergence_window: Number of epochs to average at start and end (default: 5)
    """
    problems = []
    
    if len(epoch_nums) < convergence_window * 2:
        print(f"Warning: Not enough epochs ({len(epoch_nums)}) for convergence detection with window={convergence_window}")
        return problems
    
    early_epochs = epoch_nums[:convergence_window]
    late_epochs = epoch_nums[-convergence_window:]
    
    for idx, row in df.iterrows():
        # Get mean losses for early and late epochs
        early_losses = []
        for epoch in early_epochs:
            mean_col = f'epoch_{epoch}_mean_loss'
            loss = row[mean_col]
            if not pd.isna(loss):
                early_losses.append(loss)
        
        late_losses = []
        for epoch in late_epochs:
            mean_col = f'epoch_{epoch}_mean_loss'
            loss = row[mean_col]
            if not pd.isna(loss):
                late_losses.append(loss)
        
        if len(early_losses) == 0 or len(late_losses) == 0:
            continue
        
        # Calculate average loss in early vs late epochs
        avg_early = np.mean(early_losses)
        avg_late = np.mean(late_losses)
        
        # Check if loss improved
        improvement = (avg_early - avg_late) / avg_early if avg_early > 0 else 0
        
        # Flag if:
        # 1. Loss increased (negative improvement)
        # 2. Loss decreased by less than 10% (not really converging)
        if improvement < 0:
            # Loss actually increased!
            problems.append({
                'timestep': row['timestep'],
                'position': row['position'],
                'component': row['component'],
                'gradient_type': row['gradient_type'],
                'issue': 'Loss increasing',
                'details': f'Avg loss increased from {avg_early:.6f} (epochs {early_epochs[0]}-{early_epochs[-1]}) to {avg_late:.6f} (epochs {late_epochs[0]}-{late_epochs[-1]})',
                'severity': 'CRITICAL'
            })
        elif improvement < 0.1:  # Less than 10% improvement
            # Loss barely decreased or plateaued
            severity = 'HIGH' if improvement < 0.01 else 'MEDIUM'
            problems.append({
                'timestep': row['timestep'],
                'position': row['position'],
                'component': row['component'],
                'gradient_type': row['gradient_type'],
                'issue': 'Not converging',
                'details': f'Loss only improved by {improvement*100:.1f}%: {avg_early:.6f} → {avg_late:.6f} (epochs {early_epochs[0]}-{early_epochs[-1]} → {late_epochs[0]}-{late_epochs[-1]})',
                'severity': severity
            })
    
    return problems


def detect_high_variance(df, epoch_nums, var_threshold=0.01, check_window=5):
    """Detect losses with consistently high variance in later epochs
    
    Args:
        var_threshold: Variance threshold
        check_window: Number of final epochs to check (default: 5)
    """
    problems = []
    
    if len(epoch_nums) < check_window:
        print(f"Warning: Not enough epochs ({len(epoch_nums)}) for variance detection with window={check_window}")
        return problems
    
    # Check variance in the last few epochs (should be stable by then)
    late_epochs = epoch_nums[-check_window:]
    
    for idx, row in df.iterrows():
        variances = []
        for epoch in late_epochs:
            var_col = f'epoch_{epoch}_var_loss'
            var = row[var_col]
            if not pd.isna(var):
                variances.append(var)
        
        if len(variances) == 0:
            continue
        
        # Check average variance in late epochs
        avg_var = np.mean(variances)
        max_var = np.max(variances)
        
        if avg_var > var_threshold:
            problems.append({
                'timestep': row['timestep'],
                'position': row['position'],
                'component': row['component'],
                'gradient_type': row['gradient_type'],
                'issue': 'High variance',
                'details': f'Avg variance in late epochs ({late_epochs[0]}-{late_epochs[-1]}): {avg_var:.6f} > {var_threshold} (max: {max_var:.6f})',
                'severity': 'MEDIUM'
            })
    
    return problems


def detect_early_stopping_opportunity(df, epoch_nums, plateau_window=5, min_improvement=0.001):
    """Detect if loss plateaued (stopped improving) - could have stopped training earlier
    
    Args:
        plateau_window: Number of consecutive epochs to check for plateau (default: 5)
        min_improvement: Minimum improvement per epoch to consider it still learning (default: 0.001)
    """
    problems = []
    
    if len(epoch_nums) < plateau_window:
        return problems
    
    for idx, row in df.iterrows():
        # Get all mean losses
        losses = []
        for epoch in epoch_nums:
            mean_col = f'epoch_{epoch}_mean_loss'
            loss = row[mean_col]
            if not pd.isna(loss):
                losses.append((epoch, loss))
        
        if len(losses) < plateau_window:
            continue
        
        # Check for plateaus: look for windows where loss barely changes
        # Start checking after first few epochs (give it time to start learning)
        start_check = max(3, len(losses) // 4)  # Start checking after 25% of training
        
        for i in range(start_check, len(losses) - plateau_window + 1):
            window_losses = [loss for _, loss in losses[i:i+plateau_window]]
            
            # Calculate improvement in this window
            window_start = window_losses[0]
            window_end = window_losses[-1]
            window_improvement = (window_start - window_end) / window_start if window_start > 0 else 0
            
            # If improvement is very small, it plateaued
            if window_improvement < min_improvement:
                # Find when it actually stopped improving (go backwards to find best epoch)
                best_epoch = i
                best_loss = window_start
                for j in range(i-1, -1, -1):
                    if losses[j][1] < best_loss:
                        best_loss = losses[j][1]
                        best_epoch = losses[j][0]
                    else:
                        break
                
                problems.append({
                    'timestep': row['timestep'],
                    'position': row['position'],
                    'component': row['component'],
                    'gradient_type': row['gradient_type'],
                    'issue': 'Loss plateaued (early stopping opportunity)',
                    'details': f'Loss plateaued at epoch {losses[i][0]}: {window_start:.6f} → {window_end:.6f} (improvement: {window_improvement*100:.2f}%). Best was epoch {best_epoch} ({best_loss:.6f})',
                    'severity': 'LOW'
                })
                break  # Only report first plateau
    
    return problems


def detect_loss_spikes(df, epoch_nums, spike_threshold=0.1):
    """Detect sudden large increases in loss (spikes)
    
    Args:
        spike_threshold: Relative increase threshold (default: 0.1 = 10% increase)
    """
    problems = []
    
    if len(epoch_nums) < 2:
        return problems
    
    for idx, row in df.iterrows():
        # Get all mean losses
        losses = []
        for epoch in epoch_nums:
            mean_col = f'epoch_{epoch}_mean_loss'
            loss = row[mean_col]
            if not pd.isna(loss):
                losses.append((epoch, loss))
        
        if len(losses) < 2:
            continue
        
        # Check for spikes: compare each epoch to previous
        for i in range(1, len(losses)):
            prev_epoch, prev_loss = losses[i-1]
            curr_epoch, curr_loss = losses[i]
            
            # Skip if previous loss is very small (can't compute meaningful ratio)
            if prev_loss < 1e-6:
                continue
            
            # Calculate relative increase
            relative_increase = (curr_loss - prev_loss) / prev_loss
            
            if relative_increase > spike_threshold:
                problems.append({
                    'timestep': row['timestep'],
                    'position': row['position'],
                    'component': row['component'],
                    'gradient_type': row['gradient_type'],
                    'issue': 'Loss spike',
                    'details': f'Loss spiked at epoch {curr_epoch}: {prev_loss:.6f} → {curr_loss:.6f} (increase: {relative_increase*100:.1f}%)',
                    'severity': 'HIGH' if relative_increase > spike_threshold * 2 else 'MEDIUM'
                })
    
    return problems


def print_problems_summary(all_problems):
    """Print a summary of detected problems"""
    if len(all_problems) == 0:
        print("\n✓ No problems detected! All losses look healthy.")
        return
    
    print(f"\n⚠ Found {len(all_problems)} potential issues:\n")
    
    # Group by severity
    by_severity = defaultdict(list)
    for problem in all_problems:
        by_severity[problem['severity']].append(problem)
    
    # Print by severity
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if severity not in by_severity:
            continue
        
        problems = by_severity[severity]
        print(f"\n{'='*80}")
        print(f"{severity} SEVERITY: {len(problems)} issues")
        print(f"{'='*80}\n")
        
        for p in problems:
            print(f"  Timestep: {p['timestep']}, Position: {p['position']}, "
                  f"Component: {p['component']}, Gradient: {p['gradient_type']}")
            print(f"  Issue: {p['issue']}")
            print(f"  Details: {p['details']}")
            print()


def save_problems_to_csv(all_problems, output_path):
    """Save detected problems to CSV"""
    if len(all_problems) == 0:
        print(f"No problems to save.")
        return
    
    # Create directory if needed
    output_dir = Path(output_path).parent
    if output_dir and str(output_dir) != '.':
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_problems)
    
    # Reorder columns
    df = df[['severity', 'issue', 'timestep', 'position', 'component', 'gradient_type', 'details']]
    
    # Sort by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    df['severity_rank'] = df['severity'].map(severity_order)
    df = df.sort_values('severity_rank').drop('severity_rank', axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"\nProblems saved to: {output_path}")


def plot_loss_graphs(df, epoch_nums, output_dir, num_samples=10, sample_problematic=True, problems_df=None):
    """Sample keys and generate loss graphs
    
    Args:
        df: DataFrame with loss statistics
        epoch_nums: List of epoch numbers
        output_dir: Directory to save plots
        num_samples: Number of keys to sample
        sample_problematic: If True, prioritize sampling problematic keys
        problems_df: DataFrame with detected problems (if available)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select keys to plot
    if sample_problematic and problems_df is not None and len(problems_df) > 0:
        # Sample from problematic keys
        problematic_keys = problems_df[['timestep', 'position', 'component', 'gradient_type']].drop_duplicates()
        if len(problematic_keys) > 0:
            num_to_sample = min(num_samples, len(problematic_keys))
            sampled_keys = problematic_keys.sample(n=num_to_sample, random_state=42)
            print(f"\nSampling {num_to_sample} problematic keys for visualization...")
        else:
            sampled_keys = None
    else:
        sampled_keys = None
    
    # If not enough problematic keys, sample randomly from all
    if sampled_keys is None or len(sampled_keys) < num_samples:
        all_keys = df[['timestep', 'position', 'component', 'gradient_type']].drop_duplicates()
        num_random = num_samples - (len(sampled_keys) if sampled_keys is not None else 0)
        if num_random > 0:
            random_keys = all_keys.sample(n=min(num_random, len(all_keys)), random_state=42)
            if sampled_keys is not None:
                sampled_keys = pd.concat([sampled_keys, random_keys]).drop_duplicates()
            else:
                sampled_keys = random_keys
            print(f"Sampling {num_random} random keys for visualization...")
    
    # Generate plots
    for idx, key_row in sampled_keys.iterrows():
        timestep = key_row['timestep']
        position = key_row['position']
        component = key_row['component']
        gradient_type = key_row['gradient_type']
        
        # Find matching row in df
        match = df[(df['timestep'] == timestep) & 
                   (df['position'] == position) & 
                   (df['component'] == component) & 
                   (df['gradient_type'] == gradient_type)]
        
        if len(match) == 0:
            continue
        
        row = match.iloc[0]
        
        # Extract loss data
        epochs = []
        mean_losses = []
        var_losses = []
        max_losses = []
        
        for epoch in epoch_nums:
            mean_col = f'epoch_{epoch}_mean_loss'
            var_col = f'epoch_{epoch}_var_loss'
            max_col = f'epoch_{epoch}_max_loss'
            
            mean_loss = row[mean_col]
            var_loss = row[var_col]
            max_loss = row[max_col]
            
            if not pd.isna(mean_loss):
                epochs.append(epoch)
                mean_losses.append(mean_loss)
                var_losses.append(var_loss if not pd.isna(var_loss) else 0)
                max_losses.append(max_loss if not pd.isna(max_loss) else mean_loss)
        
        if len(epochs) == 0:
            continue
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Mean loss with variance as shaded region
        ax1.plot(epochs, mean_losses, 'b-', linewidth=2, label='Mean Loss')
        if any(v > 0 for v in var_losses):
            std_losses = [np.sqrt(v) if v > 0 else 0 for v in var_losses]
            ax1.fill_between(epochs, 
                            [m - s for m, s in zip(mean_losses, std_losses)],
                            [m + s for m, s in zip(mean_losses, std_losses)],
                            alpha=0.3, color='blue', label='±1 std')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Loss')
        ax1.set_title(f'Loss: Timestep={timestep}, Pos={position}, Comp={component}, Grad={gradient_type}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Max loss
        ax2.plot(epochs, max_losses, 'r-', linewidth=2, label='Max Loss', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Max Loss')
        ax2.set_title('Max Loss per Epoch')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = f"loss_ts{timestep}_pos{position}_comp{component}_grad{gradient_type}.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filepath}")
    
    print(f"\nGenerated {len(sampled_keys)} loss graphs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detect problematic losses in probing training")
    parser.add_argument("--pattern", type=str, required=True,
                       help="String pattern to search for in CSV filenames (e.g., 'sana', 'zeros')")
    parser.add_argument("--results_dir", type=str, default="probing/results_csvs",
                       help="Directory containing loss stats CSV files")
    parser.add_argument("--output_csv", type=str, default="probing/loss_problems.csv",
                       help="Output CSV file for detected problems")
    
    # Thresholds
    parser.add_argument("--convergence_window", type=int, default=5,
                       help="Number of epochs to compare at start vs end for convergence (default: 5)")
    parser.add_argument("--variance_threshold", type=float, default=0.01,
                       help="Threshold for high variance (default: 0.01)")
    parser.add_argument("--variance_check_window", type=int, default=5,
                       help="Number of final epochs to check for variance (default: 5)")
    parser.add_argument("--plateau_window", type=int, default=5,
                       help="Number of consecutive epochs to check for plateau (default: 5)")
    parser.add_argument("--min_improvement", type=float, default=0.001,
                       help="Minimum improvement per epoch to consider still learning (default: 0.001)")
    parser.add_argument("--spike_threshold", type=float, default=0.1,
                       help="Relative increase threshold for loss spikes (default: 0.1 = 10%%)")
    
    # Plotting options
    parser.add_argument("--plot_graphs", action="store_true",
                       help="Generate loss graphs for sampled keys")
    parser.add_argument("--plot_dir", type=str, default="probing/loss_graphs",
                       help="Directory to save loss graphs (default: probing/loss_graphs)")
    parser.add_argument("--num_plot_samples", type=int, default=10,
                       help="Number of keys to sample for plotting (default: 10)")
    parser.add_argument("--plot_problematic", action="store_true", default=False,
                       help="Prioritize plotting problematic keys")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LOSS PROBLEM DETECTION")
    print("=" * 80)
    print(f"Pattern: {args.pattern}")
    print(f"Results directory: {args.results_dir}")
    print(f"Settings:")
    print(f"  Convergence window: {args.convergence_window} epochs")
    print(f"  Variance threshold: {args.variance_threshold}")
    print(f"  Variance check window: {args.variance_check_window} epochs")
    print(f"  Plateau window: {args.plateau_window} epochs")
    print(f"  Min improvement: {args.min_improvement}")
    print(f"  Spike threshold: {args.spike_threshold*100:.1f}%")
    
    # Load data
    print("\n1. Loading loss statistics...")
    df = load_all_loss_stats(args.results_dir, args.pattern)
    
    # Get epoch columns
    epoch_nums = get_epoch_columns(df)
    print(f"\nFound {len(epoch_nums)} epochs: {min(epoch_nums)} to {max(epoch_nums)}")
    
    # Run detectors
    print("\n2. Running detectors...")
    all_problems = []
    
    print("  - Checking for NaN values...")
    nan_problems = detect_nan_losses(df, epoch_nums)
    all_problems.extend(nan_problems)
    print(f"    Found {len(nan_problems)} issues")
    
    print("  - Checking for non-converging losses...")
    non_converging = detect_non_converging_losses(df, epoch_nums, args.convergence_window)
    all_problems.extend(non_converging)
    print(f"    Found {len(non_converging)} issues")
    
    print("  - Checking for high variance...")
    high_var = detect_high_variance(df, epoch_nums, args.variance_threshold, args.variance_check_window)
    all_problems.extend(high_var)
    print(f"    Found {len(high_var)} issues")
    
    print("  - Checking for early stopping opportunities (plateaus)...")
    plateaus = detect_early_stopping_opportunity(df, epoch_nums, args.plateau_window, args.min_improvement)
    all_problems.extend(plateaus)
    print(f"    Found {len(plateaus)} issues")
    
    print("  - Checking for loss spikes...")
    spikes = detect_loss_spikes(df, epoch_nums, args.spike_threshold)
    all_problems.extend(spikes)
    print(f"    Found {len(spikes)} issues")
    
    # Print summary
    print("\n3. Summary:")
    print_problems_summary(all_problems)
    
    # Save to CSV
    print("\n4. Saving results...")
    save_problems_to_csv(all_problems, args.output_csv)
    
    # Generate plots if requested
    if args.plot_graphs:
        print("\n5. Generating loss graphs...")
        problems_df = pd.DataFrame(all_problems) if len(all_problems) > 0 else None
        plot_loss_graphs(df, epoch_nums, args.plot_dir, args.num_plot_samples, 
                        args.plot_problematic, problems_df)
    
    print("\n" + "=" * 80)
    print("DETECTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

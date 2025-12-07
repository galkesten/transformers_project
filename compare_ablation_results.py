#!/usr/bin/env python3
"""Compare ablation results between old and new directories."""

import os
import pandas as pd
import sys
from pathlib import Path

# Skip these combinations
SKIP_COMBINATIONS = {
    ('zero', 'cross_attn'),
    ('zero', 'self_attn'),
    ('mean_over_tokens', 'mix_ffn'),
    ('mean_over_tokens', 'cross_attn'),  # Not finished yet
}

def parse_old_filename(filename):
    """Extract ablation_type and component from old filename.
    
    Examples:
    - all_layers_results_mean_per_token_self_attn.csv -> ('mean_per_token', 'self_attn')
    - all_layers_results_zero_cross_attn.csv -> ('zero', 'cross_attn')
    """
    # Remove extension
    name = filename.replace('.csv', '')
    
    # Remove prefix
    if name.startswith('all_layers_results_'):
        name = name.replace('all_layers_results_', '')
    elif name.startswith('all_timesteps_results_'):
        name = name.replace('all_timesteps_results_', '')
        return None, None, 'step-wise'  # Skip step-wise for now, only layer-wise
    
    # Parse ablation type and component
    ablation_types = ['mean_per_token', 'mean_over_tokens', 'zero']
    components = ['self_attn', 'cross_attn', 'mix_ffn']
    
    ablation_type = None
    component = None
    
    for at in ablation_types:
        if name.startswith(at):
            ablation_type = at
            name = name.replace(at + '_', '')
            break
    
    for comp in components:
        if name == comp:
            component = comp
            break
    
    return ablation_type, component, 'layer-wise'

def build_new_path(ablation_type, component, mode='layer-wise'):
    """Build the path to the new results file."""
    base_dir = Path('ablation_results_new/quantitative_results')
    if mode == 'layer-wise':
        return base_dir / mode / ablation_type / component / f'all_layers_results_{ablation_type}_{component}.csv'
    else:
        return base_dir / 'step-wise' / ablation_type / component / f'all_timesteps_results_{ablation_type}_{component}.csv'

def compare_csv_files(old_path, new_path, tolerance=1e-6):
    """Compare two CSV files and return True if they match."""
    try:
        old_df = pd.read_csv(old_path)
        new_df = pd.read_csv(new_path)
    except Exception as e:
        print(f"  ERROR reading files: {e}")
        return False, "READ_ERROR"
    
    # Check if columns match (set comparison to ignore order)
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    if old_cols != new_cols:
        missing_in_new = old_cols - new_cols
        missing_in_old = new_cols - old_cols
        msg = "COLUMNS_DIFFER: "
        if missing_in_new:
            msg += f"missing in new: {missing_in_new}; "
        if missing_in_old:
            msg += f"missing in old: {missing_in_old}"
        return False, msg.strip()
    
    # Check if number of rows match
    if len(old_df) != len(new_df):
        return False, f"ROW_COUNT_DIFFER: old={len(old_df)}, new={len(new_df)}"
    
    # Reorder columns to match (for consistent comparison)
    common_cols = sorted(list(old_cols))
    old_df = old_df[common_cols]
    new_df = new_df[common_cols]
    
    # Compare each row
    differences = []
    for idx in range(len(old_df)):
        old_row = old_df.iloc[idx]
        new_row = new_df.iloc[idx]
        
        # Compare each column
        for col in common_cols:
            old_val = old_row[col]
            new_val = new_row[col]
            
            # Handle NaN values
            if pd.isna(old_val) and pd.isna(new_val):
                continue
            
            # Try numeric comparison first
            try:
                old_float = float(old_val)
                new_float = float(new_val)
                if abs(old_float - new_float) > tolerance:
                    differences.append(f"  Row {idx}, Col '{col}': old={old_val}, new={new_val} (diff={abs(old_float - new_float):.2e})")
            except (ValueError, TypeError):
                # String comparison
                if str(old_val) != str(new_val):
                    differences.append(f"  Row {idx}, Col '{col}': old='{old_val}', new='{new_val}'")
    
    if differences:
        return False, f"VALUES_DIFFER ({len(differences)} differences):\n" + "\n".join(differences[:15])  # Show first 15 differences
    else:
        return True, "MATCH"

def main():
    old_dir = Path('ablation_results')
    new_dir = Path('ablation_results_new/quantitative_results')
    
    if not old_dir.exists():
        print(f"ERROR: Old directory not found: {old_dir}")
        sys.exit(1)
    
    if not new_dir.exists():
        print(f"ERROR: New directory not found: {new_dir}")
        sys.exit(1)
    
    # Find all CSV files in old directory (excluding aggregated_results and sample_images)
    old_csv_files = []
    for csv_file in old_dir.glob('all_*.csv'):
        if csv_file.name.endswith('.lock'):
            continue
        old_csv_files.append(csv_file)
    
    print(f"Found {len(old_csv_files)} CSV files in {old_dir}\n")
    print("=" * 80)
    
    results = {
        'match': [],
        'differ': [],
        'missing': [],
        'skipped': [],
        'error': []
    }
    
    for old_file in sorted(old_csv_files):
        ablation_type, component, mode = parse_old_filename(old_file.name)
        
        if ablation_type is None or component is None:
            print(f"SKIP: {old_file.name} (could not parse)")
            results['skipped'].append((old_file.name, "PARSE_ERROR"))
            continue
        
        # Check if we should skip this combination
        if (ablation_type, component) in SKIP_COMBINATIONS:
            print(f"SKIP: {old_file.name} (ablation_type={ablation_type}, component={component})")
            results['skipped'].append((old_file.name, f"{ablation_type}/{component}"))
            continue
        
        if mode != 'layer-wise':
            print(f"SKIP: {old_file.name} (mode={mode}, only checking layer-wise)")
            results['skipped'].append((old_file.name, f"mode={mode}"))
            continue
        
        # Build new path
        new_file = build_new_path(ablation_type, component, mode)
        
        print(f"\nComparing:")
        print(f"  OLD: {old_file}")
        print(f"  NEW: {new_file}")
        
        if not new_file.exists():
            print(f"  RESULT: NEW file does not exist")
            results['missing'].append((old_file.name, str(new_file)))
            continue
        
        # Compare files
        match, reason = compare_csv_files(old_file, new_file)
        
        if match:
            print(f"  RESULT: ✓ MATCH")
            results['match'].append((old_file.name, str(new_file)))
        else:
            print(f"  RESULT: ✗ DIFFER - {reason}")
            results['differ'].append((old_file.name, str(new_file), reason))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files checked: {len(old_csv_files)}")
    print(f"  ✓ Matches: {len(results['match'])}")
    print(f"  ✗ Differ: {len(results['differ'])}")
    print(f"  ⊘ Missing: {len(results['missing'])}")
    print(f"  ⊘ Skipped: {len(results['skipped'])}")
    print(f"  ⚠ Errors: {len(results['error'])}")
    
    if results['differ']:
        print("\n" + "=" * 80)
        print("FILES THAT DIFFER:")
        print("=" * 80)
        for old_name, new_path, reason in results['differ']:
            print(f"\n{old_name}:")
            print(f"  {reason}")
            if "VALUES_DIFFER" in reason:
                print(f"  NOTE: If images look the same, this suggests old CSV files may have")
                print(f"        been incorrectly populated (e.g., baseline results written to")
                print(f"        ablation CSV files, or wrong evaluation files used).")
    
    if results['missing']:
        print("\n" + "=" * 80)
        print("NEW FILES MISSING:")
        print("=" * 80)
        for old_name, new_path in results['missing']:
            print(f"  {old_name} -> {new_path}")

if __name__ == '__main__':
    main()


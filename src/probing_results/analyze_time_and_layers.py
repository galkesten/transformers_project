#!/usr/bin/env python3
"""
Create visualizations for probing results:

MODE 1 (heatmap): Heatmaps with:
- X axis: timesteps (big to small, descending)
- Y axis: layers (initial at top, then 0-N, final at bottom)
- Each cell: performance value
- Separate plots for MAE and Spearman for each kernel size AND gradient type found in data

MODE 2 (lineplot): Line plots with:
- X axis: timesteps (BIG to SMALL)
- Y axis: performance metric (MAE or Spearman)
- Multiple lines: emphasizes initial/final layers + samples middle layers
- Thicker lines for initial/final layers
- Separate plots for MAE and Spearman for each kernel size AND gradient type found in data

MODE 3 (lineplot_perlayer): Line plots with:
- X axis: layers
- Y axis: performance metric (MAE or Spearman)
- Multiple lines: samples timesteps (min, max, and evenly spaced middle ones)
- Thicker lines for min/max timesteps
- Separate plots for MAE and Spearman for each kernel size AND gradient type found in data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from pathlib import Path
import glob
import argparse

# Disable HTML/MathJax rendering in kaleido for PDF export
pio.kaleido.scope.mathjax = None

def load_all_results(results_dir="probing/results_csvs", filename_pattern=None):
    """Load and concatenate CSV files matching the filename pattern
    
    Args:
        results_dir: Directory containing CSV files
        filename_pattern: String pattern to search for in filenames (e.g., "k1", "k3"). 
                         If None, loads all CSV files.
    """
    csv_files = glob.glob(f"{results_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    # Filter by filename pattern if provided
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
        print(f"Filtered to {len(csv_files)} files containing '{filename_pattern}' in filename")
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found with pattern '{filename_pattern}' in filename")
    
    # Load and merge all matching CSVs
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merging: {len(combined)}")
    
    return combined

def prepare_layer_order(df):
    """Determine layer order: initial, numeric layers (sorted), final"""
    # Get all unique positions
    positions = df['position'].unique()
    
    # Separate into initial, numeric, and final
    initial = []
    numeric_layers = []
    final = []
    
    for pos in positions:
        if pos == 'initial':
            initial.append(pos)
        elif pos == 'final':
            final.append(pos)
        else:
            # Try to convert to int
            try:
                numeric_layers.append(int(pos))
            except (ValueError, TypeError):
                # If it's already numeric, keep it
                if isinstance(pos, (int, np.integer)):
                    numeric_layers.append(int(pos))
    
    # Sort numeric layers
    numeric_layers = sorted(set(numeric_layers))
    
    # Create ordered list
    layer_order = initial + numeric_layers + final
    
    print(f"Layer order: {layer_order}")
    return layer_order

def create_heatmap(df, metric, kernel_size, gradient_type, output_dir="probing/analysis_figures"):
    """Create heatmap for a specific metric, kernel size, and gradient type"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter by kernel size and gradient type
    df_filtered = df[(df['kernel_size'] == kernel_size) & 
                     (df['gradient_type'] == gradient_type)].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for kernel_size={kernel_size}, gradient_type={gradient_type}, skipping...")
        return
    
    # Get layer order
    layer_order = prepare_layer_order(df_filtered)
    
    # Reverse layer order so initial/0 is at the top
    layer_order_reversed = list(reversed(layer_order))
    
    # Get timesteps in descending order (big to small)
    timesteps = sorted(df_filtered['timestep'].unique(), reverse=True)
    
    # Prepare data for heatmap
    # Average across components only (not gradient types - now filtered)
    if metric == 'mae':
        value_col = 'mean_mae'
        var_col = 'std_mae'  # Using std_mae as variance indicator
    elif metric == 'spearman':
        value_col = 'mean_spearman'
        var_col = 'var_spearman'  # Using var_spearman
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Convert position column to string for consistent matching
    df_filtered['position_str'] = df_filtered['position'].astype(str)
    
    # Check group sizes and print when groups have more than 1 row
    group_sizes = df_filtered.groupby(['position_str', 'timestep']).size()
    groups_with_multiple = group_sizes[group_sizes > 1]
    if len(groups_with_multiple) > 0:
        print(f"    Groups with >1 row (averaging {len(groups_with_multiple)} groups):")
        for (pos, ts), count in groups_with_multiple.items():
            print(f"      Layer {pos}, Timestep {ts}: {count} rows")
    
    # Create pivot tables for mean
    pivot_mean = df_filtered.groupby(['position_str', 'timestep'])[value_col].mean().reset_index()
    
    # Create matrices (using reversed layer order)
    mean_matrix = np.full((len(layer_order_reversed), len(timesteps)), np.nan)
    
    for i, layer in enumerate(layer_order_reversed):
        layer_str = str(layer)  # Convert layer to string for matching
        for j, ts in enumerate(timesteps):
            layer_data_mean = pivot_mean[(pivot_mean['position_str'] == layer_str) & (pivot_mean['timestep'] == ts)]
            
            if len(layer_data_mean) > 0:
                mean_matrix[i, j] = layer_data_mean[value_col].values[0]
    
    # Prepare text annotations (just mean, no variance)
    text_matrix = []
    for i in range(len(layer_order_reversed)):
        row_text = []
        for j in range(len(timesteps)):
            mean_val = mean_matrix[i, j]
            
            if not np.isnan(mean_val):
                # Format the text (just mean value)
                text = f'{mean_val:.3f}'
            else:
                text = ''
            row_text.append(text)
        text_matrix.append(row_text)
    
    # Set color scale range and colormap
    if metric == 'mae':
        vmin = np.nanmin(mean_matrix)
        vmax = np.nanmax(mean_matrix)
        colorscale = 'Reds_r'  # Reversed Reds: lower is better (darker red = lower MAE)
    else:  # spearman
        vmin = np.nanmin(mean_matrix)
        vmax = 1.0  # Use actual min but cap max at 1
        colorscale = 'Blues'  # Blues: higher is better (darker blue = higher Spearman)
    
    # Create plotly heatmap (using reversed layer order for y-axis)
    fig = go.Figure(data=go.Heatmap(
        z=mean_matrix,
        x=[str(ts) for ts in timesteps],
        y=[str(l) for l in layer_order_reversed],
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "white"},
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(
            title='Mean MAE' if metric == 'mae' else 'Mean Spearman Correlation'
        ),
        hovertemplate='Layer: %{y}\nTimestep: %{x}\nValue: %{z:.4f}<extra></extra>'
    ))
    
    # Update layout
    metric_name = 'MAE' if metric == 'mae' else 'Spearman Correlation'
    fig.update_layout(
        title=f'{metric_name} Heatmap (Kernel={kernel_size}, Grad={gradient_type})',
        xaxis_title='Timestep (big to small)',
        yaxis_title='Layer (initial at top)',
        width=max(800, len(timesteps) * 60),
        height=max(600, len(layer_order_reversed) * 40),
        xaxis=dict(tickangle=-45)
    )
    
    # Save as PNG
    filename_png = f"heatmap_{metric}_kernel{kernel_size}_grad{gradient_type}.png"
    filepath_png = f"{output_dir}/{filename_png}"
    fig.write_image(filepath_png, width=max(1200, len(timesteps) * 80), height=max(800, len(layer_order_reversed) * 50))
    
    # Save as PDF
    filename_pdf = f"heatmap_{metric}_kernel{kernel_size}_grad{gradient_type}.pdf"
    filepath_pdf = f"{output_dir}/{filename_pdf}"
    fig.write_image(filepath_pdf, width=max(1200, len(timesteps) * 80), height=max(800, len(layer_order_reversed) * 50))
    
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")

def create_lineplot(df, metric, kernel_size, gradient_type, output_dir="probing/analysis_figures"):
    """Create line plot showing selected layers across timesteps (BIG to SMALL)
    Emphasizes initial and final layers, samples a few middle layers"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter by kernel size and gradient type
    df_filtered = df[(df['kernel_size'] == kernel_size) & 
                     (df['gradient_type'] == gradient_type)].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for kernel_size={kernel_size}, gradient_type={gradient_type}, skipping...")
        return
    
    # Get layer order
    layer_order = prepare_layer_order(df_filtered)
    
    # Select layers to plot: initial, final, and sample some middle layers
    layers_to_plot = []
    numeric_layers = [l for l in layer_order if isinstance(l, int) or (isinstance(l, str) and l.isdigit())]
    
    # Always include initial and final
    if 'initial' in layer_order:
        layers_to_plot.append('initial')
    if 'final' in layer_order:
        layers_to_plot.append('final')
    
    # Sample middle numeric layers (take every 3rd layer, or adjust based on count)
    if len(numeric_layers) > 0:
        numeric_layers_sorted = sorted([int(l) if isinstance(l, str) else l for l in numeric_layers])
        # Sample approximately 5-6 middle layers
        if len(numeric_layers_sorted) <= 6:
            layers_to_plot.extend(numeric_layers_sorted)
        else:
            step = max(1, len(numeric_layers_sorted) // 5)
            sampled = numeric_layers_sorted[::step]
            layers_to_plot.extend(sampled)
    
    print(f"    Plotting layers: {layers_to_plot}")
    
    # Get timesteps in DESCENDING order (BIG to SMALL)
    timesteps = sorted(df_filtered['timestep'].unique(), reverse=True)
    
    # Prepare data
    if metric == 'mae':
        value_col = 'mean_mae'
    elif metric == 'spearman':
        value_col = 'mean_spearman'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Convert position column to string for consistent matching
    df_filtered['position_str'] = df_filtered['position'].astype(str)
    
    # Check group sizes and print when groups have more than 1 row
    group_sizes = df_filtered.groupby(['position_str', 'timestep']).size()
    groups_with_multiple = group_sizes[group_sizes > 1]
    if len(groups_with_multiple) > 0:
        print(f"    Groups with >1 row (averaging {len(groups_with_multiple)} groups):")
        for (pos, ts), count in groups_with_multiple.items():
            print(f"      Layer {pos}, Timestep {ts}: {count} rows")
    
    # Create pivot table for mean
    pivot_mean = df_filtered.groupby(['position_str', 'timestep'])[value_col].mean().reset_index()
    
    # Use full data range (no percentile clipping)
    all_values = pivot_mean[value_col].values
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range  # Add 5% padding
    y_max = y_max + 0.05 * y_range
    
    # Create figure with distinct colors for each layer
    fig = go.Figure()
    
    # Use a color palette with distinct colors
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    
    # Add a line for each selected layer, emphasizing initial and final
    for idx, layer in enumerate(layers_to_plot):
        layer_str = str(layer)
        layer_data = pivot_mean[pivot_mean['position_str'] == layer_str].sort_values('timestep', ascending=False)
        
        if len(layer_data) > 0:
            # Emphasize initial and final layers with thicker lines
            is_special = (layer == 'initial' or layer == 'final')
            line_width = 4.0 if is_special else 2.0
            marker_size = 9 if is_special else 6
            
            fig.add_trace(go.Scatter(
                x=layer_data['timestep'],
                y=layer_data[value_col],
                mode='lines+markers',
                name=f'Layer {layer}',
                line=dict(width=line_width, color=colors[idx % len(colors)]),
                marker=dict(size=marker_size),
                opacity=1.0 if is_special else 0.7
            ))
    
    # Update layout
    metric_name = 'MAE' if metric == 'mae' else 'Spearman Correlation'
    fig.update_layout(
        title=f'{metric_name} Across Timesteps (Kernel={kernel_size}, Grad={gradient_type})',
        xaxis_title='Timestep (big to small)',
        yaxis_title=metric_name,
        yaxis_range=[y_min, y_max],
        width=1200,
        height=700,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        xaxis=dict(
            autorange='reversed'  # Reverse x-axis so big values are on the left
        )
    )
    
    # Save as PNG
    filename_png = f"lineplot_{metric}_kernel{kernel_size}_grad{gradient_type}.png"
    filepath_png = f"{output_dir}/{filename_png}"
    fig.write_image(filepath_png, width=1400, height=800)
    
    # Save as PDF
    filename_pdf = f"lineplot_{metric}_kernel{kernel_size}_grad{gradient_type}.pdf"
    filepath_pdf = f"{output_dir}/{filename_pdf}"
    fig.write_image(filepath_pdf, width=1400, height=800)
    
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")

def create_lineplot_per_layer(df, metric, kernel_size, gradient_type, output_dir="probing/analysis_figures"):
    """Create line plots showing sampled timesteps as curves across layers"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter by kernel size and gradient type
    df_filtered = df[(df['kernel_size'] == kernel_size) & 
                     (df['gradient_type'] == gradient_type)].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for kernel_size={kernel_size}, gradient_type={gradient_type}, skipping...")
        return
    
    # Get layer order
    layer_order = prepare_layer_order(df_filtered)
    
    # Get all timesteps and sample them
    all_timesteps = sorted(df_filtered['timestep'].unique())
    
    # Sample timesteps: take largest, smallest, and evenly spaced middle ones
    if len(all_timesteps) <= 7:
        timesteps_to_plot = all_timesteps
    else:
        # Always include min and max
        min_ts = min(all_timesteps)
        max_ts = max(all_timesteps)
        
        # Sample approximately 5-6 middle timesteps
        step = max(1, len(all_timesteps) // 6)
        sampled = all_timesteps[::step]
        
        # Ensure min and max are included
        timesteps_to_plot = sorted(set([min_ts, max_ts] + sampled))
    
    print(f"    Plotting timesteps: {timesteps_to_plot}")
    
    # Prepare data
    if metric == 'mae':
        value_col = 'mean_mae'
    elif metric == 'spearman':
        value_col = 'mean_spearman'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Convert position column to string for consistent matching
    df_filtered['position_str'] = df_filtered['position'].astype(str)
    
    # Create pivot table for mean
    pivot_mean = df_filtered.groupby(['position_str', 'timestep'])[value_col].mean().reset_index()
    
    # Use full data range (no percentile clipping)
    all_values = pivot_mean[value_col].values
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range
    y_max = y_max + 0.05 * y_range
    
    # Use a color palette with distinct colors for timesteps
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    
    metric_name = 'MAE' if metric == 'mae' else 'Spearman Correlation'
    
    # Create a single plot with layers on x-axis and sampled timesteps as separate lines
    fig = go.Figure()
    
    for idx, ts in enumerate(timesteps_to_plot):
        ts_data = []
        layer_labels = []
        
        for layer in layer_order:
            layer_str = str(layer)
            data_point = pivot_mean[(pivot_mean['position_str'] == layer_str) & 
                                   (pivot_mean['timestep'] == ts)]
            
            if len(data_point) > 0:
                ts_data.append(data_point[value_col].values[0])
                layer_labels.append(str(layer))
        
        if len(ts_data) > 0:
            # Emphasize min and max timesteps
            is_extreme = (ts == min(timesteps_to_plot) or ts == max(timesteps_to_plot))
            line_width = 3.5 if is_extreme else 2.0
            marker_size = 8 if is_extreme else 6
            
            fig.add_trace(go.Scatter(
                x=layer_labels,
                y=ts_data,
                mode='lines+markers',
                name=f'Timestep {ts}',
                line=dict(width=line_width, color=colors[idx % len(colors)]),
                marker=dict(size=marker_size),
                opacity=1.0 if is_extreme else 0.7
            ))
    
    # Update layout
    fig.update_layout(
        title=f'{metric_name} Across Layers (Kernel={kernel_size}, Grad={gradient_type})',
        xaxis_title='Layer',
        yaxis_title=metric_name,
        yaxis_range=[y_min, y_max],
        width=1200,
        height=700,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save as PNG
    filename_png = f"lineplot_perlayer_{metric}_kernel{kernel_size}_grad{gradient_type}.png"
    filepath_png = f"{output_dir}/{filename_png}"
    fig.write_image(filepath_png, width=1400, height=800)
    
    # Save as PDF
    filename_pdf = f"lineplot_perlayer_{metric}_kernel{kernel_size}_grad{gradient_type}.pdf"
    filepath_pdf = f"{output_dir}/{filename_pdf}"
    fig.write_image(filepath_pdf, width=1400, height=800)
    
    print(f"Saved: {filepath_png}")
    print(f"Saved: {filepath_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for probing results")
    parser.add_argument("--pattern", type=str, required=True,
                       help="String pattern to search for in CSV filenames (e.g., 'k1', 'k3')")
    parser.add_argument("--results_dir", type=str, default="probing/results_csvs",
                       help="Directory containing CSV result files")
    parser.add_argument("--output_dir", type=str, default="probing/analysis_figures",
                       help="Directory to save output figures")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["heatmap", "lineplot", "lineplot_perlayer", "both", "all"],
                       help="Visualization mode: 'heatmap', 'lineplot', 'lineplot_perlayer', 'both' (heatmap+lineplot), or 'all'")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROBING RESULTS VISUALIZATION ANALYSIS")
    print("=" * 60)
    print(f"Filename pattern: {args.pattern}")
    print(f"Mode: {args.mode}")
    
    # Load data
    print("\n1. Loading data...")
    df = load_all_results(args.results_dir, args.pattern)
    
    # Check available kernel sizes and gradient types
    kernel_sizes = sorted(df['kernel_size'].unique())
    gradient_types = sorted(df['gradient_type'].unique())
    print(f"Available kernel sizes in loaded data: {kernel_sizes}")
    print(f"Available gradient types in loaded data: {gradient_types}")
    
    # Create visualizations based on mode
    print("\n2. Creating visualizations...")
    
    for kernel_size in kernel_sizes:
        for gradient_type in gradient_types:
            print(f"\n  Kernel size {kernel_size}, Gradient type {gradient_type}:")
            
            # Heatmap mode
            if args.mode in ['heatmap', 'all']:
                print(f"    - MAE heatmap...")
                create_heatmap(df, 'mae', kernel_size, gradient_type, args.output_dir)
                
                print(f"    - Spearman heatmap...")
                create_heatmap(df, 'spearman', kernel_size, gradient_type, args.output_dir)
            
            # Lineplot mode (layers as lines, timesteps on x-axis)
            if args.mode in ['lineplot', 'both', 'all']:
                print(f"    - MAE lineplot (layers across time)...")
                create_lineplot(df, 'mae', kernel_size, gradient_type, args.output_dir)
                
                print(f"    - Spearman lineplot (layers across time)...")
                create_lineplot(df, 'spearman', kernel_size, gradient_type, args.output_dir)
            
            # Lineplot per layer mode (timesteps as lines, layers on x-axis)
            if args.mode in ['lineplot_perlayer', 'all']:
                print(f"    - MAE lineplot (timesteps across layers)...")
                create_lineplot_per_layer(df, 'mae', kernel_size, gradient_type, args.output_dir)
                
                print(f"    - Spearman lineplot (timesteps across layers)...")
                create_lineplot_per_layer(df, 'spearman', kernel_size, gradient_type, args.output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All figures saved to: {args.output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()


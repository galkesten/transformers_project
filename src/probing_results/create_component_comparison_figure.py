#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
from pathlib import Path
import glob
import argparse

pio.defaults.mathjax = None

def load_component_data(results_dir="probing/results_csvs", filename_pattern=None):
    """Load component data from CSV files with 'components' in filename"""
    csv_files = glob.glob(f"{results_dir}/*.csv")
    
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
    
    # Filter for component files (containing "components" in name)
    csv_files = [f for f in csv_files if "components" in f and "loss_stats" not in f]
    
    if len(csv_files) == 0:
        raise ValueError(f"No component CSV files found with pattern '{filename_pattern}'")
    
    print(f"Found {len(csv_files)} component CSV files")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total component rows after merging: {len(combined)}")
    
    # Filter for specific components we want to compare
    desired_components = ['mix_ffn_after_gate', 'self_attn_after_gate', 'cross_attn']
    
    # Get all available component names in data
    available_components = combined['component'].unique()
    print(f"Available components in data: {sorted(available_components)}")
    
    # Build list of components to keep (check both exact match and alternatives)
    components_to_keep = []
    for comp in desired_components:
        if comp in available_components:
            components_to_keep.append(comp)
        else:
            # Check if alternative exists (e.g., mix_ffn instead of mix_ffn_after_gate)
            base_name = comp.replace('_after_gate', '')
            if base_name in available_components:
                components_to_keep.append(base_name)
    
    print(f"Components to keep: {components_to_keep}")
    
    # Filter for desired components and layer positions (numeric)
    component_data = combined[
        (combined['component'].isin(components_to_keep)) &
        (combined['position'].apply(lambda x: str(x).isdigit()))
    ].copy()
    component_data['position'] = component_data['position'].astype(int)
    print(f"Rows after filtering for components and layer positions: {len(component_data)}")
    
    return component_data

def load_block_output_data(results_dir="probing/results_csvs", filename_pattern=None):
    """Load block_output data from CSV files without 'components' in filename"""
    csv_files = glob.glob(f"{results_dir}/*.csv")
    
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
    
    # Filter for block output files (NOT containing "components", and NOT "loss_stats")
    csv_files = [f for f in csv_files if "components" not in f and "loss_stats" not in f]
    
    if len(csv_files) == 0:
        raise ValueError(f"No block_output CSV files found with pattern '{filename_pattern}'")
    
    print(f"Found {len(csv_files)} block_output CSV files")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total block_output rows after merging: {len(combined)}")
    
    # Filter for block_output component and layer positions
    block_data = combined[
        (combined['component'] == 'block_output') & 
        (combined['position'].apply(lambda x: str(x).isdigit()))
    ].copy()
    block_data['position'] = block_data['position'].astype(int)
    print(f"Rows after filtering for block_output layers: {len(block_data)}")
    
    return block_data

def load_baseline_data(results_dir="probing/results_csvs", filename_pattern=None):
    """Load baseline data from initial representation"""
    csv_files = glob.glob(f"{results_dir}/*.csv")
    
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
    
    csv_files = [f for f in csv_files if "loss_stats" not in f]
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found with pattern '{filename_pattern}'")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Filter for initial representation
    initial_data = combined[
        (combined['position'] == 'initial') & 
        (combined['component'] == 'patch_embed')
    ].copy()
    
    if len(initial_data) == 0:
        print("Warning: No initial representation data found!")
        return {}
    
    # Get max_timestep for baseline
    all_timesteps = combined[combined['timestep'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]['timestep'].unique()
    max_timestep = int(max(all_timesteps)) if len(all_timesteps) > 0 else None
    
    if max_timestep is None:
        print("Warning: Could not determine max_timestep!")
        return {}
    
    baseline_values = {}
    for grad_type in initial_data['gradient_type'].unique():
        baseline_data = initial_data[
            (initial_data['gradient_type'] == grad_type) & 
            (initial_data['timestep'] == max_timestep)
        ]
        if len(baseline_data) > 0:
            baseline_values[grad_type] = {
                'mae': baseline_data['mean_mae'].iloc[0],
                'spearman': baseline_data['mean_spearman'].iloc[0],
                'timestep': max_timestep
            }
            print(f"Baseline for {grad_type} from timestep {max_timestep}: MAE={baseline_values[grad_type]['mae']:.4f}, Spearman={baseline_values[grad_type]['spearman']:.4f}")
    
    return baseline_values

def map_component_name(component):
    """Map component names to display names"""
    mapping = {
        'mix_ffn_after_gate': 'Mix-FFN',
        'self_attn_after_gate': 'Self-Attn',
        'cross_attn': 'Cross-Attn',
        'block_output': 'Block(Residual Stream)'
    }
    return mapping.get(component, component)

def apply_smoothing(values, smooth, window=3):
    """Apply rolling average smoothing to values if requested"""
    if smooth and len(values) >= window:
        return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values
    return values

def get_component_data(df, timestep, grad_type, component):
    """Get and sort component data for a specific timestep and gradient type"""
    comp_data = df[
        (df['timestep'] == timestep) &
        (df['gradient_type'] == grad_type) &
        (df['component'] == component)
    ].copy()
    
    if len(comp_data) > 0:
        comp_data = comp_data.sort_values('position')
    
    return comp_data

def plot_component_traces(
    fig, comp_data, display_name, color, smooth,
    mae_row, mae_col, spearman_row, spearman_col,
    showlegend, secondary_y_mae=None, secondary_y_spearman=None,
    use_solid_for_spearman=False
):
    """Plot MAE and Spearman traces for a component
    
    Args:
        use_solid_for_spearman: If True, use solid line for Spearman (for single timestep mode)
    """
    if len(comp_data) == 0:
        return
    
    layers = comp_data['position'].values
    mae_values = apply_smoothing(comp_data['mean_mae'].values, smooth)
    spearman_values = apply_smoothing(comp_data['mean_spearman'].values, smooth)
    
    # MAE trace (solid line)
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=mae_values,
            mode='lines',
            name=display_name,
            line=dict(color=color, width=2, dash='solid'),
            showlegend=showlegend,
            legendgroup=display_name,
            hoverinfo='skip'
        ),
        row=mae_row, col=mae_col, 
        secondary_y=secondary_y_mae
    )
    
    # Spearman trace (solid or dashed line depending on mode)
    spearman_dash = 'solid' if use_solid_for_spearman else 'dash'
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=spearman_values,
            mode='lines',
            name=display_name,
            line=dict(color=color, width=2, dash=spearman_dash),
            showlegend=False,
            legendgroup=display_name,
            hoverinfo='skip'
        ),
        row=spearman_row, col=spearman_col,
        secondary_y=secondary_y_spearman
    )

def plot_single_timestep_mode(
    fig, timestep, gradient_types, all_components, component_colors,
    all_data_df, smooth
):
    """Plot single timestep with 2 rows (MAE, Spearman) x 3 cols (Gradient types)"""
    
    # Plot for each gradient type
    for col_idx, grad_type in enumerate(gradient_types):
        col = col_idx + 1
        
        # Plot all components (including block_output)
        for comp in all_components:
            comp_data = get_component_data(all_data_df, timestep, grad_type, comp)
            display_name = map_component_name(comp)
            color = component_colors[display_name]
            
            plot_component_traces(
                fig, comp_data, display_name, color, smooth,
                mae_row=1, mae_col=col,
                spearman_row=2, spearman_col=col,
                showlegend=(col_idx == 0),
                use_solid_for_spearman=True  # Use solid lines for single timestep mode
            )
        
        # Update axes
        fig.update_xaxes(title_text='Layer', row=1, col=col)
        fig.update_xaxes(title_text='Layer', row=2, col=col)
        fig.update_yaxes(title_text='MAE', row=1, col=col)
        fig.update_yaxes(title_text='Spearman Correlation', row=2, col=col)

def plot_dual_yaxis_mode(
    fig, timesteps, gradient_types, all_components, component_colors,
    all_data_df, smooth
):
    """Plot with dual y-axis (MAE left, Spearman right) in same subplot"""
    
    # Plot data for each timestep and gradient type
    for row_idx, timestep in enumerate(timesteps):
        for col_idx, grad_type in enumerate(gradient_types):
            row = row_idx + 1
            col = col_idx + 1
            
            # Plot all components (including block_output)
            for comp in all_components:
                comp_data = get_component_data(all_data_df, timestep, grad_type, comp)
                display_name = map_component_name(comp)
                color = component_colors[display_name]
                
                plot_component_traces(
                    fig, comp_data, display_name, color, smooth,
                    mae_row=row, mae_col=col,
                    spearman_row=row, spearman_col=col,
                    showlegend=(row_idx == 0 and col_idx == 0),
                    secondary_y_mae=False,
                    secondary_y_spearman=True
                )
            
            # Update axes
            fig.update_xaxes(title_text='Layer', row=row, col=col)
            fig.update_yaxes(title_text='MAE', row=row, col=col, secondary_y=False)
            fig.update_yaxes(title_text='Spearman Correlation', row=row, col=col, secondary_y=True)

def plot_two_subplots_mode(
    fig, timesteps, gradient_types, all_components, component_colors,
    all_data_df, smooth
):
    """Plot with two separate subplots side-by-side (MAE left, Spearman right)"""
    
    # Plot data for each timestep and gradient type
    for row_idx, timestep in enumerate(timesteps):
        for col_idx, grad_type in enumerate(gradient_types):
            row = row_idx + 1
            mae_col = col_idx * 2 + 1  # MAE subplot
            spearman_col = col_idx * 2 + 2  # Spearman subplot
            
            # Plot all components (including block_output)
            for comp in all_components:
                comp_data = get_component_data(all_data_df, timestep, grad_type, comp)
                display_name = map_component_name(comp)
                color = component_colors[display_name]
                
                plot_component_traces(
                    fig, comp_data, display_name, color, smooth,
                    mae_row=row, mae_col=mae_col,
                    spearman_row=row, spearman_col=spearman_col,
                    showlegend=(row_idx == 0 and col_idx == 0)
                )
            
            # Update axes
            fig.update_xaxes(title_text='Layer', row=row, col=mae_col)
            fig.update_xaxes(title_text='Layer', row=row, col=spearman_col)
            fig.update_yaxes(title_text='MAE', row=row, col=mae_col)
            fig.update_yaxes(title_text='Spearman Correlation', row=row, col=spearman_col)

def plot_component_comparison(
    component_df, 
    block_df, 
    baseline_values,
    output_dir="probing/figures",
    kernel_size=None,
    start_timestep_index=0,
    timestep_step=1,
    dual_y_axis=False,
    filename_suffix="",
    smooth=False,
    include_last_timestep=False,
    single_timestep=None
):
    """Create component comparison figure"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter by kernel_size
    if kernel_size is not None:
        component_df = component_df[component_df['kernel_size'] == kernel_size].copy()
        block_df = block_df[block_df['kernel_size'] == kernel_size].copy()
        print(f"Filtered to kernel_size={kernel_size}")
    
    if len(component_df) == 0 or len(block_df) == 0:
        print("No data to plot after filtering!")
        return
    
    # Get all timesteps sorted from highest to lowest (for iteration numbering)
    all_timesteps_sorted = sorted(component_df['timestep'].unique(), reverse=True)
    
    # Check if single_timestep mode is requested
    if single_timestep is not None:
        # Single timestep mode: use only the specified timestep
        if single_timestep not in all_timesteps_sorted:
            print(f"Warning: Timestep {single_timestep} not found in data. Available timesteps: {all_timesteps_sorted[:5]}...")
            return
        timesteps = [single_timestep]
        # Calculate iteration number (1-indexed: 999 is iteration 1)
        timestep_iteration = all_timesteps_sorted.index(single_timestep) + 1
        print(f"Single timestep mode: using t={single_timestep} (iteration {timestep_iteration})")
    else:
        # Get all timesteps and sample them
        # Fix: Use simple slicing to include start point each time
        timesteps = all_timesteps_sorted[start_timestep_index::timestep_step] if timestep_step > 1 else all_timesteps_sorted[start_timestep_index:]
        
        # Optionally include the last timestep if not already included
        if include_last_timestep and len(all_timesteps_sorted) > 0:
            last_timestep = all_timesteps_sorted[-1]  # Last timestep (smallest value, reverse sorted)
            if last_timestep not in timesteps:
                timesteps = list(timesteps) + [last_timestep]
                # Re-sort to maintain descending order
                timesteps = sorted(timesteps, reverse=True)
        
        timestep_iteration = None  # Not used in multi-timestep mode
    
    gradient_types = sorted(component_df['gradient_type'].unique())
    layers = sorted(component_df['position'].unique())
    
    # Combine component_df and block_df for unified processing
    all_data_df = pd.concat([component_df, block_df], ignore_index=True)
    
    # Determine which components are available in the data
    available_components = set(component_df['component'].unique())
    desired_components = ['mix_ffn_after_gate', 'self_attn_after_gate', 'cross_attn']
    components_to_plot = [c for c in desired_components if c in available_components]
    
    # Add block_output to components list
    all_components = components_to_plot + ['block_output']
    
    print(f"Timesteps to plot: {timesteps} ({len(timesteps)} total)")
    print(f"Gradient types: {gradient_types}")
    print(f"Layers: {layers}")
    print(f"All components to plot: {all_components}")
    
    # Component colors (consistent across all plots) - avoid similar colors
    component_colors = {
        'Mix-FFN': px.colors.qualitative.Plotly[0],  # Blue
        'Self-Attn': px.colors.qualitative.Plotly[1],  # Red
        'Cross-Attn': px.colors.qualitative.Plotly[2],  # Green
        'Block(Residual Stream)': px.colors.qualitative.Plotly[4]  # Orange (skip purple [3])
    }
    
    # Collect all MAE and Spearman values to determine global ranges for axis alignment
    all_mae_values = []
    all_spearman_values = []
    
    # Collect from all components data
    for timestep in timesteps:
        for grad_type in gradient_types:
            for comp in all_components:
                comp_data = get_component_data(all_data_df, timestep, grad_type, comp)
                if len(comp_data) > 0:
                    all_mae_values.extend(comp_data['mean_mae'].values)
                    all_spearman_values.extend(comp_data['mean_spearman'].values)
    
    # Calculate global ranges with small padding
    if len(all_mae_values) > 0:
        mae_min = min(all_mae_values)
        mae_max = max(all_mae_values)
        mae_range = mae_max - mae_min
        mae_padding = mae_range * 0.05  # 5% padding
        mae_y_range = [max(0, mae_min - mae_padding), mae_max + mae_padding]
    else:
        mae_y_range = None
    
    if len(all_spearman_values) > 0:
        spearman_min = min(all_spearman_values)
        spearman_max = max(all_spearman_values)
        spearman_range = spearman_max - spearman_min
        spearman_padding = spearman_range * 0.05  # 5% padding
        spearman_y_range = [max(-1, spearman_min - spearman_padding), min(1, spearman_max + spearman_padding)]
    else:
        spearman_y_range = None
    
    print(f"Global MAE range: {mae_y_range}")
    print(f"Global Spearman range: {spearman_y_range}")
    
    # Create figure
    # Define num_rows and num_cols for layout calculations
    if single_timestep is not None:
        num_rows = 2  # MAE row and Spearman row
        num_cols = len(gradient_types)  # One column per gradient type
    else:
        num_rows = len(timesteps)
        num_cols = len(gradient_types)
    
    if single_timestep is not None:
        print(f"Single timestep mode: {single_timestep}")
        # Single timestep mode: 2 rows (MAE, Spearman) x 3 cols (Gradient types)
        # Titles: Row 1 = MAE for all gradients, Row 2 = Spearman for all gradients
        subplot_titles = []
        for grad_type in gradient_types:
            subplot_titles.append(f"<b>{grad_type} - MAE</b>")
        for grad_type in gradient_types:
            subplot_titles.append(f"<b>{grad_type} - Spearman</b>")
        
        fig = make_subplots(
            rows=2,
            cols=len(gradient_types),
            subplot_titles=subplot_titles,
            vertical_spacing=0.25,  # Same as create_figures.py for more space between rows
            horizontal_spacing=0.10  # Same as create_figures.py
        )
        
        # Call helper function to plot
        plot_single_timestep_mode(
            fig, single_timestep, gradient_types, all_components, component_colors,
            all_data_df, smooth
        )
    
    elif dual_y_axis:
        # Dual y-axis mode
        subplot_titles = []
        for ts in timesteps:
            # Calculate iteration number for this timestep
            iter_num = all_timesteps_sorted.index(ts) + 1
            for grad_type in gradient_types:
                subplot_titles.append(f"t={ts} (iter {iter_num}), {grad_type}")
        
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True} for _ in range(num_cols)] for _ in range(num_rows)],
            vertical_spacing=0.10,
            horizontal_spacing=0.15  # Increased from 0.08 to 0.15 for more space
        )
        
        plot_dual_yaxis_mode(
            fig, timesteps, gradient_types, all_components, component_colors,
            all_data_df, smooth
        )
    
    elif not dual_y_axis and single_timestep is None:
        # Two subplots mode
        subplot_titles = []
        for ts in timesteps:
            # Calculate iteration number for this timestep
            iter_num = all_timesteps_sorted.index(ts) + 1
            for grad_type in gradient_types:
                subplot_titles.append(f"t={ts} (iter {iter_num}), {grad_type} - MAE")
                subplot_titles.append(f"t={ts} (iter {iter_num}), {grad_type} - Spearman")
        
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols * 2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.10,
            horizontal_spacing=0.05
        )
        
        plot_two_subplots_mode(
            fig, timesteps, gradient_types, all_components, component_colors,
            all_data_df, smooth
        )
    
    # Add line style indicators to legend (to show solid = MAE, dashed = Spearman)
    # Only add these for multi-timestep modes (not single timestep mode where MAE and Spearman are separated)
    if single_timestep is None:
        # Add dummy traces that won't show on plot but appear in legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name='Solid line',
                line=dict(color='black', width=2, dash='solid'),
                showlegend=True,
                legendgroup='line_styles',
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name='Dashed line',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=True,
                legendgroup='line_styles',
                hoverinfo='skip'
            )
        )
    
    # Update layout with increased font sizes (matching create_figures.py style)
    if single_timestep is not None:
        layout_height = 500  # Reduced height to make it wider/more landscape
        layout_width = 1000  # Keep width same as create_figures.py
        top_margin = 40  # No title, so less top margin needed
    else:
        layout_height = 400 * num_rows
        layout_width = 1400 if dual_y_axis else 2800
        top_margin = 60
    
    layout_config = {
        'height': layout_height,
        'width': layout_width,
        'plot_bgcolor': '#e6f3ff',
        'paper_bgcolor': 'white',
        'font': dict(size=14, family="Times New Roman"),  # Increased from default
        'showlegend': True,
        'legend': dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,  # Reduced from -0.35 to bring legend closer
            xanchor='center',
            x=0.5,
            font=dict(size=14, family="Times New Roman"),  # Increased from 16 to match style
            traceorder='normal',
            itemwidth=30
        ),
        'margin': dict(l=60, r=40, t=top_margin, b=100)  # Reduced bottom margin from 120
    }
    
    fig.update_layout(**layout_config)
    
    # Align all MAE and Spearman axes to have the same ranges
    if mae_y_range is not None:
        if single_timestep is not None:
            # For single timestep mode: set MAE range on row 1 (MAE row)
            for col in range(1, len(gradient_types) + 1):
                fig.update_yaxes(range=mae_y_range, row=1, col=col)
        elif dual_y_axis:
            # For dual y-axis mode: set MAE range on left y-axis (secondary_y=False)
            for row in range(1, num_rows + 1):
                for col in range(1, num_cols + 1):
                    fig.update_yaxes(range=mae_y_range, row=row, col=col, secondary_y=False)
        else:
            # For two subplots mode: set MAE range on MAE subplots
            for row in range(1, num_rows + 1):
                for col_idx in range(len(gradient_types)):
                    mae_col = col_idx * 2 + 1
                    fig.update_yaxes(range=mae_y_range, row=row, col=mae_col)
    
    if spearman_y_range is not None:
        if single_timestep is not None:
            # For single timestep mode: set Spearman range on row 2 (Spearman row)
            for col in range(1, len(gradient_types) + 1):
                fig.update_yaxes(range=spearman_y_range, row=2, col=col)
        elif dual_y_axis:
            # For dual y-axis mode: set Spearman range on right y-axis (secondary_y=True)
            for row in range(1, num_rows + 1):
                for col in range(1, num_cols + 1):
                    fig.update_yaxes(range=spearman_y_range, row=row, col=col, secondary_y=True)
        else:
            # For two subplots mode: set Spearman range on Spearman subplots
            for row in range(1, num_rows + 1):
                for col_idx in range(len(gradient_types)):
                    spearman_col = col_idx * 2 + 2
                    fig.update_yaxes(range=spearman_y_range, row=row, col=spearman_col)
    
    # Update axis label font sizes (matching create_figures.py style)
    fig.update_xaxes(
        title_font=dict(size=16, family="Times New Roman"),
        tickfont=dict(size=14, family="Times New Roman")
    )
    fig.update_yaxes(
        title_font=dict(size=16, family="Times New Roman"),
        tickfont=dict(size=14, family="Times New Roman")
    )
    
    # Update subplot title font sizes (matching create_figures.py)
    if single_timestep is not None:
        # For single timestep mode, update all subplot title annotations
        for annotation in fig.layout.annotations:
            if annotation.text:
                annotation['font'] = dict(size=12, family="Times New Roman")  # Same as create_figures.py
    
    # Save figure
    kernel_str = f"_kernel{kernel_size}" if kernel_size is not None else ""
    if single_timestep is not None:
        mode_str = f"_singletimestep{single_timestep}"
    else:
        dual_str = "_dualyaxis" if dual_y_axis else "_twosubplots"
        step_str = f"_tstep{timestep_step}" if timestep_step > 1 else ""
        mode_str = f"{dual_str}{step_str}"
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    
    filename_png = f"component_comparison{kernel_str}{mode_str}{suffix_str}.png"
    filepath = Path(output_dir) / filename_png
    fig.write_image(str(filepath), width=fig.layout.width, height=fig.layout.height)
    print(f"Saved: {filepath}")
    
    filename_pdf = f"component_comparison{kernel_str}{mode_str}{suffix_str}.pdf"
    filepath_pdf = Path(output_dir) / filename_pdf
    fig.write_image(str(filepath_pdf), width=fig.layout.width, height=fig.layout.height)
    print(f"Saved: {filepath_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Create component comparison figure")
    parser.add_argument("--results_dir", type=str, default="probing/results_csvs",
                       help="Directory containing CSV result files")
    parser.add_argument("--output_dir", type=str, default="probing/figures",
                       help="Directory to save output figures")
    parser.add_argument("--pattern", type=str, default="ln_conv_sana",
                       help="String pattern to search for in CSV filenames")
    parser.add_argument("--kernel_size", type=int, default=1,
                       help="Filter by kernel size")
    parser.add_argument("--start_timestep_index", type=int, default=0,
                       help="Starting index for timestep selection")
    parser.add_argument("--timestep_step", type=int, default=1,
                       help="Show every Xth timestep by index")
    parser.add_argument("--dual_y_axis", action="store_true",
                       help="Use dual y-axis in same subplot (default: two separate subplots)")
    parser.add_argument("--suffix", type=str, default="",
                       help="Optional suffix to add to output filename")
    parser.add_argument("--smooth", action="store_true",
                       help="Apply 3-point rolling average smoothing to all curves")
    parser.add_argument("--include_last_timestep", action="store_true",
                       help="Include the last timestep (smallest value) even if not in sampled range")
    parser.add_argument("--single_timestep", type=int, default=None,
                       help="Single timestep mode: plot one timestep with MAE row and Spearman row (2x3 layout)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CREATING COMPONENT COMPARISON FIGURE")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Filename pattern: {args.pattern}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Timestep start index: {args.start_timestep_index}")
    print(f"Timestep step: {args.timestep_step}")
    print(f"Dual y-axis mode: {args.dual_y_axis}")
    print(f"Smoothing: {args.smooth}")
    print(f"Include last timestep: {args.include_last_timestep}")
    print(f"Single timestep mode: {args.single_timestep}")
    
    # Load data
    print("\n1. Loading component data...")
    component_df = load_component_data(args.results_dir, args.pattern)
    
    print("\n2. Loading block_output data...")
    block_df = load_block_output_data(args.results_dir, args.pattern)
    
    print("\n3. Loading baseline data...")
    baseline_values = load_baseline_data(args.results_dir, args.pattern)
    print(f"Baseline values: {baseline_values}")
    
    print("\n4. Creating component comparison figure...")
    plot_component_comparison(
        component_df,
        block_df,
        baseline_values,
        output_dir=args.output_dir,
        kernel_size=args.kernel_size,
        start_timestep_index=args.start_timestep_index,
        timestep_step=args.timestep_step,
        dual_y_axis=args.dual_y_axis,
        filename_suffix=args.suffix,
        smooth=args.smooth,
        include_last_timestep=args.include_last_timestep,
        single_timestep=args.single_timestep
    )
    
    print("\n" + "=" * 60)
    print("FIGURE CREATED!")
    print(f"Figure saved to: {args.output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()


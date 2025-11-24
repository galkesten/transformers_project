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

pio.kaleido.scope.mathjax = None

def load_final_representation_data(results_dir="probing/results_csvs", filename_pattern=None):
    csv_files = glob.glob(f"{results_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
        print(f"Filtered to {len(csv_files)} files containing '{filename_pattern}' in filename")
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found with pattern '{filename_pattern}' in filename")
    
    csv_files = [f for f in csv_files if "loss_stats" not in f]
    print(f"After excluding loss_stats files: {len(csv_files)} files remaining")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merging: {len(combined)}")
    
    final_data = combined[(combined['position'] == 'final') & (combined['component'] == 'proj_out')].copy()
    print(f"Rows after filtering for final representation: {len(final_data)}")
    
    initial_data = combined[(combined['position'] == 'initial') & (combined['component'] == 'patch_embed')].copy()
    print(f"Rows after filtering for initial representation: {len(initial_data)}")
    
    return final_data, initial_data

def plot_final_representation_mae_spearman(df, initial_df, output_dir="probing/figures", kernel_size=None, filename_suffix="", include_initial_curves=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if kernel_size is not None:
        df = df[df['kernel_size'] == kernel_size].copy()
        initial_df = initial_df[initial_df['kernel_size'] == kernel_size].copy()
        print(f"Filtered to kernel_size={kernel_size}: {len(df)} rows")
    
    if len(df) == 0:
        print("No data to plot after filtering!")
        return
    
    gradient_types = sorted(df['gradient_type'].unique())
    print(f"Gradient types found: {gradient_types}")
    
    timesteps = sorted(df['timestep'].unique(), reverse=True)
    print(f"Timesteps: {timesteps[:5]}...{timesteps[-2:]} ({len(timesteps)} total)")
    
    max_timestep = max(timesteps) if len(timesteps) > 0 else None
    
    baseline_values = {}
    if max_timestep is not None:
        for grad_type in gradient_types:
            baseline_data = initial_df[
                (initial_df['gradient_type'] == grad_type) & 
                (initial_df['timestep'] == max_timestep)
            ]
            if len(baseline_data) > 0:
                baseline_values[grad_type] = {
                    'mae': baseline_data['mean_mae'].iloc[0],
                    'spearman': baseline_data['mean_spearman'].iloc[0],
                    'timestep': max_timestep
                }
                print(f"Baseline for {grad_type} from timestep {max_timestep}: MAE={baseline_values[grad_type]['mae']:.4f}, Spearman={baseline_values[grad_type]['spearman']:.4f}")
    
    colors = {
        'Vertical': px.colors.qualitative.Plotly[0],
        'Horizontal': px.colors.qualitative.Plotly[1],
        'Gaussian': px.colors.qualitative.Plotly[2]
    }
    
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12
    )
    
    for grad_type in gradient_types:
        grad_data = df[df['gradient_type'] == grad_type].copy()
        
        grad_data = grad_data.sort_values('timestep', ascending=False)
        
        timesteps_grad = grad_data['timestep'].values
        mae_values = grad_data['mean_mae'].values
        spearman_values = grad_data['mean_spearman'].values
        mae_std = grad_data['std_mae'].values
        spearman_var = grad_data['var_spearman'].values
        spearman_std = np.sqrt(spearman_var)
        
        color = colors.get(grad_type, '#808080')
        
        hover_label = f'{grad_type} (final)' if include_initial_curves else grad_type
        
        fig.add_trace(
            go.Scatter(
                x=timesteps_grad,
                y=mae_values,
                mode='lines+markers',
                name=grad_type,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                showlegend=False,
                legendgroup=grad_type,
                hovertemplate=f'<b>{hover_label}</b><br>Timestep: %{{x}}<br>MAE: %{{y:.4f}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timesteps_grad,
                y=spearman_values,
                mode='lines+markers',
                name=grad_type,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                showlegend=False,
                legendgroup=grad_type,
                hovertemplate=f'<b>{hover_label}</b><br>Timestep: %{{x}}<br>Spearman: %{{y:.4f}}<extra></extra>'
            ),
            row=1, col=2
        )
    
    if include_initial_curves:
        for grad_type in gradient_types:
            initial_grad_data = initial_df[initial_df['gradient_type'] == grad_type].copy()
            
            if len(initial_grad_data) == 0:
                continue
            
            initial_grad_data = initial_grad_data.sort_values('timestep', ascending=False)
            
            timesteps_initial = initial_grad_data['timestep'].values
            mae_values_initial = initial_grad_data['mean_mae'].values
            spearman_values_initial = initial_grad_data['mean_spearman'].values
            
            color = colors.get(grad_type, '#808080')
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps_initial,
                    y=mae_values_initial,
                    mode='lines+markers',
                    name=f'{grad_type} (initial)',
                    line=dict(color=color, width=2.5, dash='dot'),
                    marker=dict(size=4, color=color, symbol='diamond'),
                    showlegend=False,
                    legendgroup=f'{grad_type}_initial',
                    hovertemplate=f'<b>{grad_type} (initial)</b><br>Timestep: %{{x}}<br>MAE: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps_initial,
                    y=spearman_values_initial,
                    mode='lines+markers',
                    name=f'{grad_type} (initial)',
                    line=dict(color=color, width=2.5, dash='dot'),
                    marker=dict(size=4, color=color, symbol='diamond'),
                    showlegend=False,
                    legendgroup=f'{grad_type}_initial',
                    hovertemplate=f'<b>{grad_type} (initial)</b><br>Timestep: %{{x}}<br>Spearman: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=2
            )
    
    if not include_initial_curves:
        for grad_type in gradient_types:
            if grad_type in baseline_values:
                color = colors.get(grad_type, '#808080')
                baseline_mae = baseline_values[grad_type]['mae']
                baseline_spearman = baseline_values[grad_type]['spearman']
                
                fig.add_trace(
                    go.Scatter(
                        x=[1050, -50],
                        y=[baseline_mae, baseline_mae],
                        mode='lines',
                        name=f'{grad_type} baseline',
                        line=dict(color=color, width=2.5, dash='dash'),
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[1050, -50],
                        y=[baseline_spearman, baseline_spearman],
                        mode='lines',
                        name=f'{grad_type} baseline',
                        line=dict(color=color, width=2.5, dash='dash'),
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
    
    for grad_type in gradient_types:
        color = colors.get(grad_type, '#808080')
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines+markers',
                name=grad_type,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
                showlegend=True,
                legendgroup=f'legend_{grad_type}',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    if include_initial_curves:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines+markers',
                name='Final',
                line=dict(color='gray', width=2.5),
                marker=dict(size=5, color='gray'),
                showlegend=True,
                legendgroup='legend_final',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines+markers',
                name='Initial',
                line=dict(color='gray', width=2.5, dash='dot'),
                marker=dict(size=4, color='gray', symbol='diamond'),
                showlegend=True,
                legendgroup='legend_initial',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    mae_all_values = []
    spearman_all_values = []
    
    for grad_type in gradient_types:
        grad_data = df[df['gradient_type'] == grad_type].copy()
        mae_all_values.extend(grad_data['mean_mae'].values.tolist())
        spearman_all_values.extend(grad_data['mean_spearman'].values.tolist())
        
        if include_initial_curves:
            initial_grad_data = initial_df[initial_df['gradient_type'] == grad_type].copy()
            mae_all_values.extend(initial_grad_data['mean_mae'].values.tolist())
            spearman_all_values.extend(initial_grad_data['mean_spearman'].values.tolist())
        elif grad_type in baseline_values:
            mae_all_values.append(baseline_values[grad_type]['mae'])
            spearman_all_values.append(baseline_values[grad_type]['spearman'])
    
    mae_min = min(mae_all_values) if mae_all_values else 0
    mae_max = max(mae_all_values) if mae_all_values else 1
    mae_range = mae_max - mae_min
    mae_padding = mae_range * 0.08
    
    spearman_min = min(spearman_all_values) if spearman_all_values else 0
    spearman_max = max(spearman_all_values) if spearman_all_values else 1
    spearman_range = spearman_max - spearman_min
    spearman_padding = spearman_range * 0.08
    
    fig.update_xaxes(
        title_text='Timestep',
        range=[1050, -50],
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Timestep',
        range=[1050, -50],
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text='MAE',
        range=[mae_min - mae_padding, mae_max + mae_padding],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Spearman Correlation',
        range=[spearman_min - spearman_padding, spearman_max + spearman_padding],
        row=1, col=2
    )
    
    light_blue = '#e6f3ff'
    
    fig.update_layout(
        height=400,
        width=800,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=11, family="Times New Roman"),
            traceorder='normal'
        ),
        hovermode='x unified',
        plot_bgcolor=light_blue,
        paper_bgcolor='white',
        margin=dict(l=50, r=30, t=20, b=70),
        font=dict(size=11, family="Times New Roman"),
        autosize=False
    )
    
    fig.update_xaxes(
        gridcolor='white',
        zeroline=False,
        showgrid=True,
        gridwidth=1,
        title_font=dict(size=12, family="Times New Roman"),
        tickfont=dict(size=10, family="Times New Roman")
    )
    fig.update_yaxes(
        gridcolor='white',
        zeroline=False,
        showgrid=True,
        gridwidth=1,
        title_font=dict(size=12, family="Times New Roman"),
        tickfont=dict(size=10, family="Times New Roman")
    )
    
    kernel_str = f"_kernel{kernel_size}" if kernel_size is not None else ""
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    representation_str = "final_initial_representation" if include_initial_curves else "final_representation"
    
    filename = f"{representation_str}_mae_spearman{kernel_str}{suffix_str}.png"
    filepath = Path(output_dir) / filename
    fig.write_image(str(filepath), width=800, height=400)
    print(f"Saved: {filepath}")
    
    filename_pdf = f"{representation_str}_mae_spearman{kernel_str}{suffix_str}.pdf"
    filepath_pdf = Path(output_dir) / filename_pdf
    fig.write_image(str(filepath_pdf), width=800, height=400)
    print(f"Saved: {filepath_pdf}")

def load_layer_data(results_dir="probing/results_csvs", filename_pattern=None):
    csv_files = glob.glob(f"{results_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    if filename_pattern:
        csv_files = [f for f in csv_files if filename_pattern in f]
        print(f"Filtered to {len(csv_files)} files containing '{filename_pattern}' in filename")
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found with pattern '{filename_pattern}' in filename")
    
    csv_files = [f for f in csv_files if "loss_stats" not in f]
    print(f"After excluding loss_stats files: {len(csv_files)} files remaining")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded: {Path(f).name} ({len(df)} rows)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merging: {len(combined)}")
    
    layer_data = combined[combined['position'].apply(lambda x: str(x).isdigit())].copy()
    layer_data['position'] = layer_data['position'].astype(int)
    print(f"Rows after filtering for layer positions: {len(layer_data)}")
    
    initial_data = combined[(combined['position'] == 'initial') & (combined['component'] == 'patch_embed')].copy()
    print(f"Rows after filtering for initial representation: {len(initial_data)}")
    
    return layer_data, initial_data

def plot_layer_aggregation(df, initial_df, output_dir="probing/figures", kernel_size=None, layer_step=1, start_layer_index=0, filename_suffix="", smooth=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if kernel_size is not None:
        df = df[df['kernel_size'] == kernel_size].copy()
        initial_df = initial_df[initial_df['kernel_size'] == kernel_size].copy()
        print(f"Filtered to kernel_size={kernel_size}: {len(df)} rows")
    
    if len(df) == 0:
        print("No data to plot after filtering!")
        return
    
    df['layer_num'] = df['position']
    
    gradient_types = sorted(df['gradient_type'].unique())
    timesteps = sorted(df['timestep'].unique(), reverse=True)
    all_layers = sorted(df['layer_num'].unique())
    
    layers = [all_layers[i] for i in range(len(all_layers)) if i >= start_layer_index and (i - start_layer_index) % layer_step == 0]
    
    print(f"Gradient types: {gradient_types}")
    print(f"Timesteps: {timesteps[:5]}...{timesteps[-2:]} ({len(timesteps)} total)")
    print(f"All layers: {all_layers} ({len(all_layers)} total)")
    print(f"Starting at index {start_layer_index}, showing every {layer_step} layer(s): {layers} ({len(layers)} layers)")
    
    max_timestep = max(timesteps) if len(timesteps) > 0 else None
    
    baseline_values = {}
    if max_timestep is not None:
        for grad_type in gradient_types:
            baseline_data = initial_df[
                (initial_df['gradient_type'] == grad_type) & 
                (initial_df['timestep'] == max_timestep)
            ]
            if len(baseline_data) > 0:
                baseline_values[grad_type] = {
                    'mae': baseline_data['mean_mae'].iloc[0],
                    'spearman': baseline_data['mean_spearman'].iloc[0],
                    'timestep': max_timestep
                }
                print(f"Baseline for {grad_type} from timestep {max_timestep}: MAE={baseline_values[grad_type]['mae']:.4f}, Spearman={baseline_values[grad_type]['spearman']:.4f}")
    
    colors = px.colors.qualitative.Plotly[:len(layers)]
    if len(layers) > len(colors):
        colors = px.colors.sample_colorscale("Viridis", [i/(len(layers)-1) for i in range(len(layers))])
    
    subplot_titles = []
    for grad_type in gradient_types:
        subplot_titles.append(f"<b>{grad_type} - MAE</b>")
        subplot_titles.append(f"<b>{grad_type} - Spearman</b>")
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.12
    )
    
    for grad_idx, grad_type in enumerate(gradient_types):
        grad_data = df[df['gradient_type'] == grad_type].copy()
        
        row = grad_idx + 1
        
        for layer_idx, layer in enumerate(layers):
            layer_data = grad_data[grad_data['layer_num'] == layer].copy()
            
            if len(layer_data) == 0:
                continue
            
            layer_data = layer_data.sort_values('timestep', ascending=False)
            
            timesteps_layer = layer_data['timestep'].values
            mae_values = layer_data['mean_mae'].values
            spearman_values = layer_data['mean_spearman'].values
            
            if smooth:
                smooth_window = 3
                if len(mae_values) >= smooth_window:
                    mae_values_plot = pd.Series(mae_values).rolling(window=smooth_window, center=True, min_periods=1).mean().values
                    spearman_values_plot = pd.Series(spearman_values).rolling(window=smooth_window, center=True, min_periods=1).mean().values
                else:
                    mae_values_plot = mae_values
                    spearman_values_plot = spearman_values
                line_shape = 'spline'
            else:
                mae_values_plot = mae_values
                spearman_values_plot = spearman_values
                line_shape = 'linear'
            
            color = colors[layer_idx % len(colors)]
            layer_label = f"Layer {layer}"
            
            show_legend = (grad_idx == 0)
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps_layer,
                    y=mae_values_plot,
                    mode='lines',
                    name=layer_label,
                    line=dict(color=color, width=1.5, shape=line_shape),
                    showlegend=show_legend,
                    legendgroup=f'layer_{layer}',
                    hovertemplate=f'<b>{layer_label}</b><br>Timestep: %{{x}}<br>MAE: %{{y:.4f}}<extra></extra>'
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps_layer,
                    y=spearman_values_plot,
                    mode='lines',
                    name=layer_label,
                    line=dict(color=color, width=1.5, shape=line_shape),
                    showlegend=False,
                    legendgroup=f'layer_{layer}',
                    hovertemplate=f'<b>{layer_label}</b><br>Timestep: %{{x}}<br>Spearman: %{{y:.4f}}<extra></extra>'
                ),
                row=row, col=2
            )
    
    baseline_color_map = {
        'Vertical': px.colors.qualitative.Plotly[0],
        'Horizontal': px.colors.qualitative.Plotly[1],
        'Gaussian': px.colors.qualitative.Plotly[2]
    }
    
    for grad_idx, grad_type in enumerate(gradient_types):
        if grad_type in baseline_values:
            row = grad_idx + 1
            color = baseline_color_map.get(grad_type, '#808080')
            baseline_mae = baseline_values[grad_type]['mae']
            baseline_spearman = baseline_values[grad_type]['spearman']
            
            fig.add_trace(
                go.Scatter(
                    x=[1050, -50],
                    y=[baseline_mae, baseline_mae],
                    mode='lines',
                    name=f'{grad_type} baseline',
                    line=dict(color=color, width=2.5, dash='dash'),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[1050, -50],
                    y=[baseline_spearman, baseline_spearman],
                    mode='lines',
                    name=f'{grad_type} baseline',
                    line=dict(color=color, width=2.5, dash='dash'),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=2
            )
    
    for row_idx in range(1, 4):
        fig.update_xaxes(title_text='Timestep', range=[1050, -50], row=row_idx, col=1)
        fig.update_xaxes(title_text='Timestep', range=[1050, -50], row=row_idx, col=2)
        fig.update_yaxes(title_text='MAE', row=row_idx, col=1)
        fig.update_yaxes(title_text='Spearman Correlation', row=row_idx, col=2)
    
    fig.update_layout(
        height=1200,
        width=1000,
        plot_bgcolor='#e6f3ff',
        paper_bgcolor='white',
        font=dict(size=11, family="Times New Roman"),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        margin=dict(l=60, r=40, t=60, b=140)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, family="Times New Roman")
    
    kernel_str = f"_kernel{kernel_size}" if kernel_size is not None else ""
    step_str = f"_step{layer_step}" if layer_step > 1 else ""
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    
    filename_png = f"layer_performance{kernel_str}{step_str}{suffix_str}.png"
    filepath = Path(output_dir) / filename_png
    fig.write_image(str(filepath), width=1000, height=1200)
    print(f"Saved: {filepath}")
    
    filename_pdf = f"layer_performance{kernel_str}{step_str}{suffix_str}.pdf"
    filepath_pdf = Path(output_dir) / filename_pdf
    fig.write_image(str(filepath_pdf), width=1000, height=1200)
    print(f"Saved: {filepath_pdf}")

def plot_timestep_across_layers(df, initial_df, output_dir="probing/figures", kernel_size=None, timestep_step=1, start_timestep_index=0, filename_suffix="", smooth=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if kernel_size is not None:
        df = df[df['kernel_size'] == kernel_size].copy()
        initial_df = initial_df[initial_df['kernel_size'] == kernel_size].copy()
        print(f"Filtered to kernel_size={kernel_size}: {len(df)} rows")
    
    if len(df) == 0:
        print("No data to plot after filtering!")
        return
    
    df['layer_num'] = df['position']
    
    gradient_types = sorted(df['gradient_type'].unique())
    layers = sorted(df['layer_num'].unique())
    all_timesteps = sorted(df['timestep'].unique(), reverse=True)
    
    timesteps = [all_timesteps[i] for i in range(len(all_timesteps)) if i >= start_timestep_index and (i - start_timestep_index) % timestep_step == 0]
    
    print(f"Gradient types: {gradient_types}")
    print(f"Layers: {layers} ({len(layers)} total)")
    print(f"All timesteps: {all_timesteps[:5]}...{all_timesteps[-2:]} ({len(all_timesteps)} total)")
    print(f"Starting at index {start_timestep_index}, showing every {timestep_step} index(es): {timesteps[:5]}...{timesteps[-2:] if len(timesteps) > 2 else timesteps} ({len(timesteps)} timesteps)")
    
    max_timestep = max(all_timesteps) if len(all_timesteps) > 0 else None
    
    baseline_values = {}
    if max_timestep is not None:
        for grad_type in gradient_types:
            baseline_data = initial_df[
                (initial_df['gradient_type'] == grad_type) & 
                (initial_df['timestep'] == max_timestep)
            ]
            if len(baseline_data) > 0:
                baseline_values[grad_type] = {
                    'mae': baseline_data['mean_mae'].iloc[0],
                    'spearman': baseline_data['mean_spearman'].iloc[0],
                    'timestep': max_timestep
                }
                print(f"Baseline for {grad_type} from timestep {max_timestep}: MAE={baseline_values[grad_type]['mae']:.4f}, Spearman={baseline_values[grad_type]['spearman']:.4f}")
    
    if len(timesteps) > 1:
        colors = px.colors.sample_colorscale("Viridis", [i/(len(timesteps)-1) for i in range(len(timesteps))])
    else:
        colors = [px.colors.qualitative.Plotly[0]]
    
    subplot_titles = []
    for grad_type in gradient_types:
        subplot_titles.append(f"<b>{grad_type} - MAE</b>")
    for grad_type in gradient_types:
        subplot_titles.append(f"<b>{grad_type} - Spearman</b>")
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    for grad_idx, grad_type in enumerate(gradient_types):
        grad_data = df[df['gradient_type'] == grad_type].copy()
        
        col = grad_idx + 1
        
        for timestep_idx, timestep in enumerate(timesteps):
            timestep_data = grad_data[grad_data['timestep'] == timestep].copy()
            
            if len(timestep_data) == 0:
                continue
            
            timestep_data = timestep_data.sort_values('layer_num')
            
            layers_t = timestep_data['layer_num'].values
            mae_values = timestep_data['mean_mae'].values
            spearman_values = timestep_data['mean_spearman'].values
            
            if smooth:
                smooth_window = 3
                if len(mae_values) >= smooth_window:
                    mae_values_plot = pd.Series(mae_values).rolling(window=smooth_window, center=True, min_periods=1).mean().values
                    spearman_values_plot = pd.Series(spearman_values).rolling(window=smooth_window, center=True, min_periods=1).mean().values
                else:
                    mae_values_plot = mae_values
                    spearman_values_plot = spearman_values
                line_shape = 'spline'
            else:
                mae_values_plot = mae_values
                spearman_values_plot = spearman_values
                line_shape = 'linear'
            
            color = colors[timestep_idx % len(colors)]
            timestep_label = f"t={timestep}"
            
            show_legend = (grad_idx == 0)
            
            fig.add_trace(
                go.Scatter(
                    x=layers_t,
                    y=mae_values_plot,
                    mode='lines',
                    name=timestep_label,
                    line=dict(color=color, width=1.5, shape=line_shape),
                    showlegend=show_legend,
                    legendgroup=f'timestep_{timestep}',
                    hovertemplate=f'<b>{timestep_label}</b><br>Layer: %{{x}}<br>MAE: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=layers_t,
                    y=spearman_values_plot,
                    mode='lines',
                    name=timestep_label,
                    line=dict(color=color, width=1.5, shape=line_shape),
                    showlegend=False,
                    legendgroup=f'timestep_{timestep}',
                    hovertemplate=f'<b>{timestep_label}</b><br>Layer: %{{x}}<br>Spearman: %{{y:.4f}}<extra></extra>'
                ),
                row=2, col=col
            )
    
    baseline_color_map = {
        'Vertical': px.colors.qualitative.Plotly[0],
        'Horizontal': px.colors.qualitative.Plotly[1],
        'Gaussian': px.colors.qualitative.Plotly[2]
    }
    
    for grad_idx, grad_type in enumerate(gradient_types):
        if grad_type in baseline_values:
            col = grad_idx + 1
            color = baseline_color_map.get(grad_type, '#808080')
            baseline_mae = baseline_values[grad_type]['mae']
            baseline_spearman = baseline_values[grad_type]['spearman']
            
            layer_min = min(layers)
            layer_max = max(layers)
            
            fig.add_trace(
                go.Scatter(
                    x=[layer_min, layer_max],
                    y=[baseline_mae, baseline_mae],
                    mode='lines',
                    name=f'{grad_type} baseline',
                    line=dict(color=color, width=2.5, dash='dash'),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[layer_min, layer_max],
                    y=[baseline_spearman, baseline_spearman],
                    mode='lines',
                    name=f'{grad_type} baseline',
                    line=dict(color=color, width=2.5, dash='dash'),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=col
            )
    
    for col_idx in range(1, 4):
        fig.update_xaxes(title_text='Layer', row=1, col=col_idx)
        fig.update_xaxes(title_text='Layer', row=2, col=col_idx)
        fig.update_yaxes(title_text='MAE', row=1, col=col_idx)
        fig.update_yaxes(title_text='Spearman Correlation', row=2, col=col_idx)
    
    fig.update_layout(
        height=800,
        width=1400,
        plot_bgcolor='#e6f3ff',
        paper_bgcolor='white',
        font=dict(size=11, family="Times New Roman"),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.22,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        margin=dict(l=60, r=40, t=60, b=160)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, family="Times New Roman")
    
    kernel_str = f"_kernel{kernel_size}" if kernel_size is not None else ""
    step_str = f"_tstep{timestep_step}" if timestep_step > 1 else ""
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    
    filename_png = f"timestep_across_layers{kernel_str}{step_str}{suffix_str}.png"
    filepath = Path(output_dir) / filename_png
    fig.write_image(str(filepath), width=1400, height=800)
    print(f"Saved: {filepath}")
    
    filename_pdf = f"timestep_across_layers{kernel_str}{step_str}{suffix_str}.pdf"
    filepath_pdf = Path(output_dir) / filename_pdf
    fig.write_image(str(filepath_pdf), width=1400, height=800)
    print(f"Saved: {filepath_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Create figures for probing results")
    parser.add_argument("--results_dir", type=str, default="probing/results_csvs",
                       help="Directory containing CSV result files")
    parser.add_argument("--output_dir", type=str, default="probing/figures",
                       help="Directory to save output figures")
    parser.add_argument("--pattern", type=str, default=None,
                       help="String pattern to search for in CSV filenames (e.g., 'k1', 'sana')")
    parser.add_argument("--kernel_size", type=int, default=None,
                       help="Filter by kernel size (optional)")
    parser.add_argument("--suffix", type=str, default="",
                       help="Optional suffix to add to output filename")
    parser.add_argument("--include_initial_curves", action="store_true",
                       help="Include initial representation curves alongside final (instead of just baseline)")
    parser.add_argument("--mode", type=str, choices=['final', 'layers', 'timesteps', 'all'], default='final',
                       help="Which figure to create: 'final' (final representation), 'layers' (layer performance over time), 'timesteps' (timestep performance across layers), or 'all'")
    parser.add_argument("--layer_step", type=int, default=1,
                       help="Show every Xth layer (e.g., 2 shows layers 0,2,4,6...; 1 shows all layers)")
    parser.add_argument("--start_layer_index", type=int, default=0,
                       help="Starting index for layer selection (e.g., 2 with step=3 shows layers at indices 2,5,8...)")
    parser.add_argument("--timestep_step", type=int, default=1,
                       help="Show every Xth timestep by index (e.g., 5 shows indices 0,5,10,15... from big to small; 1 shows all)")
    parser.add_argument("--start_timestep_index", type=int, default=0,
                       help="Starting index for timestep selection (e.g., 2 with step=3 shows timesteps at indices 2,5,8...)")
    parser.add_argument("--smooth", action="store_true",
                       help="Apply smoothing to curves (rolling average + spline interpolation)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CREATING PROBING RESULTS FIGURES")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Filename pattern: {args.pattern}")
    print(f"Kernel size filter: {args.kernel_size}")
    print(f"Mode: {args.mode}")
    
    if args.mode in ['final', 'all']:
        print("\n1. Loading final representation data...")
        df, initial_df = load_final_representation_data(args.results_dir, args.pattern)
        
        if len(df) == 0:
            print("No final representation data found!")
        else:
            kernel_sizes = sorted(df['kernel_size'].unique())
            print(f"Available kernel sizes: {kernel_sizes}")
            
            print("\n2. Creating final representation figure...")
            if args.kernel_size is not None:
                plot_final_representation_mae_spearman(df, initial_df, args.output_dir, 
                                                      kernel_size=args.kernel_size,
                                                      filename_suffix=args.suffix,
                                                      include_initial_curves=args.include_initial_curves)
            else:
                for k in kernel_sizes:
                    print(f"\n  Creating figure for kernel_size={k}...")
                    plot_final_representation_mae_spearman(df, initial_df, args.output_dir, 
                                                          kernel_size=k,
                                                          filename_suffix=args.suffix,
                                                          include_initial_curves=args.include_initial_curves)
    
    if args.mode in ['layers', 'all']:
        print("\n3. Loading layer data...")
        layer_df, layer_initial_df = load_layer_data(args.results_dir, args.pattern)
        
        if len(layer_df) == 0:
            print("No layer data found!")
        else:
            kernel_sizes = sorted(layer_df['kernel_size'].unique())
            print(f"Available kernel sizes: {kernel_sizes}")
            
            print("\n4. Creating layer performance figure...")
            if args.kernel_size is not None:
                plot_layer_aggregation(layer_df, layer_initial_df, args.output_dir,
                                      kernel_size=args.kernel_size,
                                      layer_step=args.layer_step,
                                      start_layer_index=args.start_layer_index,
                                      filename_suffix=args.suffix,
                                      smooth=args.smooth)
            else:
                for k in kernel_sizes:
                    print(f"\n  Creating figure for kernel_size={k}...")
                    plot_layer_aggregation(layer_df, layer_initial_df, args.output_dir,
                                          kernel_size=k,
                                          layer_step=args.layer_step,
                                          start_layer_index=args.start_layer_index,
                                          filename_suffix=args.suffix,
                                          smooth=args.smooth)
    
    if args.mode in ['timesteps', 'all']:
        print("\n5. Loading layer data for timestep analysis...")
        timestep_df, timestep_initial_df = load_layer_data(args.results_dir, args.pattern)
        
        if len(timestep_df) == 0:
            print("No layer data found!")
        else:
            kernel_sizes = sorted(timestep_df['kernel_size'].unique())
            print(f"Available kernel sizes: {kernel_sizes}")
            
            print("\n6. Creating timestep across layers figure...")
            if args.kernel_size is not None:
                plot_timestep_across_layers(timestep_df, timestep_initial_df, args.output_dir,
                                           kernel_size=args.kernel_size,
                                           timestep_step=args.timestep_step,
                                           start_timestep_index=args.start_timestep_index,
                                           filename_suffix=args.suffix,
                                           smooth=args.smooth)
            else:
                for k in kernel_sizes:
                    print(f"\n  Creating figure for kernel_size={k}...")
                    plot_timestep_across_layers(timestep_df, timestep_initial_df, args.output_dir,
                                               kernel_size=k,
                                               timestep_step=args.timestep_step,
                                               start_timestep_index=args.start_timestep_index,
                                               filename_suffix=args.suffix,
                                               smooth=args.smooth)
    
    print("\n" + "=" * 60)
    print("FIGURES CREATED!")
    print(f"All figures saved to: {args.output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()

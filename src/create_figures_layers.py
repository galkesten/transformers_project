import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re
import sys
import os

def extract_layer_num(layer_str):
    m = re.search(r'(\d+)', str(layer_str))
    return int(m.group(1)) if m else -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--metric', type=str, required=True, help='Metric column to plot (e.g., mae, spearman)')
    parser.add_argument('--out_folder', type=str, required=True, help='Output folder to save the figure')
    parser.add_argument('--kernel', type=int, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.out_folder):
        print(f"Output folder '{args.out_folder}' does not exist.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        print(f"Metric '{args.metric}' not found in columns: {df.columns.tolist()}")
        sys.exit(1)

    df['layer_num'] = df['layer'].apply(extract_layer_num)
    df = df.sort_values('layer_num')

    gradient_types = df['gradient_type'].unique()
    df = df[df['kernel'] == args.kernel]
    components = df['component'].unique()

    fig, axes = plt.subplots(1, len(gradient_types), figsize=(6*len(gradient_types), 5), sharey=True)
    if len(gradient_types) == 1:
        axes = [axes]

    for ax, grad in zip(axes, gradient_types):
        grad_df = df[df['gradient_type'] == grad]
        for comp in components:
            subset = grad_df[(grad_df['kernel'] == args.kernel) & (grad_df['component'] == comp)]
            if subset.empty:
                continue
            ax.plot(
                subset['layer_num'],
                subset[args.metric],
                marker='o',
                label=f'Kernel {args.kernel}, {comp}'
                )
        ax.set_title(f"Gradient: {grad}")
        ax.set_xlabel("Layer")
        ax.set_xticks(subset['layer_num'])
        ax.set_xticklabels(subset['layer'], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True)

    axes[0].set_ylabel(args.metric)
    if args.metric.lower() == "mae":
        all_values = df[args.metric].values
        ymin = all_values.min()
        ymax = all_values.max()
        ymargin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1 * ymax
        for ax in axes:
            ax.set_ylim(ymin - ymargin, ymax + ymargin)
    elif args.metric.lower() == "spearman":
        for ax in axes:
            ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.suptitle(f"{args.metric} across Layers (by Gradient, Kernel, Component)", y=1.04, fontsize=16)

    out_path = os.path.join(args.out_folder, f"{args.metric}_by_layer.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    main()

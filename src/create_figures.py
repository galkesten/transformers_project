import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--x', type=str, required=True, help='Column to use for x-axis')
    parser.add_argument('--y', type=str, required=True, help='Column to use for y-axis')
    parser.add_argument('--title_col', type=str, required=True, help='Column to use for subplot titles')
    parser.add_argument('--line_group_col', type=str, default='kernel', help='Column to use for line grouping/color (default: kernel)')
    parser.add_argument('--line_group_values', type=str, nargs='*', default=None, help='Values of line_group_col to plot (default: all)')
    parser.add_argument('--title_values', type=str, nargs='*', default=None, help='Which values of title_col to use as subplots (default: all)')
    parser.add_argument('--out', type=str, required=True, help='Output image file')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    # Sanity check
    for col in [args.x, args.y, args.title_col, args.line_group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV columns: {df.columns.tolist()}")

    # Split the dataframe: main lines and baselines
    df_lines = df[df[args.x] != 0]
    df_baseline = df[df[args.x] == 0]

    # Values for subplots and lines
    if args.title_values is not None:
        subplot_values = args.title_values
    else:
        subplot_values = sorted(df[args.title_col].unique())

    if args.line_group_values is not None:
        line_values = args.line_group_values
    else:
        line_values = sorted(df[args.line_group_col].unique())

    # Color map for up to 10 lines
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = {v: color_list[i % len(color_list)] for i, v in enumerate(line_values)}

    # Create output dir if needed
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axs = plt.subplots(1, len(subplot_values), figsize=(6 * len(subplot_values), 5), sharey=True)

    # If only one subplot, axs is not a list
    if len(subplot_values) == 1:
        axs = [axs]

    for i, title_val in enumerate(subplot_values):
        ax = axs[i]
        for v in line_values:
            # Plot main lines (non-baseline)
            d = df_lines[(df_lines[args.title_col] == title_val) & (df_lines[args.line_group_col] == v)]
            d_sorted = d.sort_values(args.x)
            if not d_sorted.empty:
                ax.plot(
                    d_sorted[args.x],
                    d_sorted[args.y],
                    marker='o',
                    label=f'{args.line_group_col}={v}',
                    color=colors[v]
                )
            # Plot baseline if exists (as a diamond, not connected)
            d_base = df_baseline[(df_baseline[args.title_col] == title_val) & (df_baseline[args.line_group_col] == v)]
            if not d_base.empty:
                # place at just left of minimum x value for this group
                min_x = d_sorted[args.x].min() if not d_sorted.empty else df_lines[args.x].min()
                baseline_x = min_x - 0.05 * (df_lines[args.x].max() - df_lines[args.x].min())
                ax.scatter(
                    [baseline_x],
                    d_base[args.y],
                    marker='D',
                    s=80,
                    color=colors[v],
                    label=f'baseline for {args.line_group_col}={v}'
                )
        ax.set_title(str(title_val))
        ax.set_xlabel(args.x)
        ax.invert_xaxis()
        if i == 0:
            ax.set_ylabel(args.y)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(args.out)
    plt.close(fig)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

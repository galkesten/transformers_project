import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--x', type=str, required=True, help='Column to use for x-axis')
    parser.add_argument('--y', type=str, required=True, help='Column to use for y-axis')
    parser.add_argument('--title_col', type=str, required=True, help='Column to use for subplot titles')
    parser.add_argument('--line_group_col', type=str, default='kernel', help='Column to use for line grouping/color')
    parser.add_argument('--line_group_values', type=str, nargs='*', default=None, help='Values of line_group_col to plot')
    parser.add_argument('--title_values', type=str, nargs='*', default=None, help='Which values of title_col to use as subplots')
    parser.add_argument('--out', type=str, required=True, help='Output image file')
    parser.add_argument('--invert_x', action='store_true', help='Invert the x-axis')
    parser.add_argument('--no_legend', action='store_true', help='Do not show legend')
    parser.add_argument('--super_title', type=str, default=None, help='Supertitle for the plot')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    for col in [args.x, args.y, args.title_col, args.line_group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV columns: {df.columns.tolist()}")

    df_lines = df[df[args.x] != 0]
    df_baseline = df[df[args.x] == 0]

    subplot_values = args.title_values or sorted(df[args.title_col].unique())
    line_values = args.line_group_values or sorted(df[args.line_group_col].unique())

    cmap = cm.get_cmap('tab10') if len(line_values) <= 10 else cm.get_cmap('tab20')
    colors = {v: cmap(i % cmap.N) for i, v in enumerate(line_values)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axs = plt.subplots(
        1, len(subplot_values),
        figsize=(6 * len(subplot_values), 5),
        sharey=True
    )

    if len(subplot_values) == 1:
        axs = [axs]

    handles_all = []
    labels_all = []

    for i, title_val in enumerate(subplot_values):
        ax = axs[i]
        for v in line_values:
            d = df_lines[(df_lines[args.title_col] == title_val) & (df_lines[args.line_group_col] == v)]
            d_sorted = d.sort_values(args.x)
            if not d_sorted.empty:
                h, = ax.plot(
                    d_sorted[args.x],
                    d_sorted[args.y],
                    marker='o',
                    label=f'{args.line_group_col}={v}',
                    color=colors[v],
                    linewidth=2
                )
                if f'{args.line_group_col}={v}' not in labels_all:
                    handles_all.append(h)
                    labels_all.append(f'{args.line_group_col}={v}')
            # Baseline: dashed horizontal line ONLY (no diamond, no text)
            d_base = df_baseline[(df_baseline[args.title_col] == title_val) & (df_baseline[args.line_group_col] == v)]
            if not d_base.empty:
                for yval in d_base[args.y]:
                    ax.axhline(
                        y=yval,
                        color=colors[v],
                        linestyle='dashed',
                        linewidth=1.7,
                        alpha=0.8,
                        zorder=1
                    )
        ax.set_title(str(title_val), fontsize=14)
        ax.set_xlabel(args.x, fontsize=13)
        if args.invert_x:
            ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Only one y-label
    if len(subplot_values) > 1:
        fig.supylabel(args.y, fontsize=13)
    else:
        axs[0].set_ylabel(args.y, fontsize=13)

    # Shared legend
    if not args.no_legend and handles_all:
        fig.legend(
            handles_all, labels_all,
            loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(line_values),
            fontsize=12, frameon=False
        )

    if args.super_title:
        fig.suptitle(args.super_title, fontsize=16, y=1.15)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

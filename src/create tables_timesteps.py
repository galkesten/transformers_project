import pandas as pd

def aggregate_tables(mae_csv, spearman_csv, kernels=(1, 3)):
    df_mae = pd.read_csv(mae_csv)
    df_spearman = pd.read_csv(spearman_csv)
    
    timesteps = [999, 982, 963, 944, 922, 899, 874, 847, 817, 785, 749, 710, 666, 617,
                 562, 499, 428, 345, 249, 136]

    # build labels: 999 -> "999 (iter 1)", ..., plus 0 -> "baseline"
    timestep_to_label = {0: "baseline"}
    timestep_to_label.update({t: f"{t} (iter {i+1})" for i, t in enumerate(timesteps)})

    key_cols = ['timestep', 'gradient_type', 'kernel']
    for col in key_cols:
        if col not in df_mae.columns or col not in df_spearman.columns:
            raise ValueError(f"Missing column '{col}' in one of the input CSVs.")

    merged = pd.merge(
        df_mae[key_cols + ['mae']],
        df_spearman[key_cols + ['spearman']],
        on=key_cols,
        how='inner'
    )

    for k in kernels:
        df_k = merged[merged['kernel'] == k].copy()
        # sort by gradient_type then numeric timestep (desc)
        df_k = df_k.sort_values(['gradient_type', 'timestep'], ascending=[True, False])

        # blank repeated gradient_type for readability
        df_k['gradient_type'] = df_k['gradient_type'].mask(
            df_k['gradient_type'] == df_k['gradient_type'].shift(), ''
        )

        # map timestep -> label; fallback to raw str if not in mapping
        df_k['timestep_label'] = df_k['timestep'].map(timestep_to_label).fillna(df_k['timestep'].astype(str))

        table = df_k.set_index(['gradient_type', 'timestep_label'])[['mae', 'spearman']]
        table_formatted = table.applymap(lambda x: f"{x:.3f}")

        print(f"\n=== Aggregated Table for kernel={k} ===")
        print(table_formatted)

        # write the labeled version
        table_formatted.to_csv(f"agg_mae_spearman_kernel{k}.csv")

# Example call:
aggregate_tables(
    "sana_outputs/results_mae_post_layer_norm_latents.csv",
    "sana_outputs/results_spearman_post_layer_norm_latents.csv",
    kernels=(1, 3)
)

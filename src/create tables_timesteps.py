
import pandas as pd

def aggregate_tables(mae_csv, spearman_csv, kernels=(1, 3)):
    df_mae = pd.read_csv(mae_csv)
    df_spearman = pd.read_csv(spearman_csv)
    
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
        df_k = df_k.sort_values(['gradient_type', 'timestep'], ascending=[True, False])
        df_k['gradient_type'] = df_k['gradient_type'].mask(df_k['gradient_type'] == df_k['gradient_type'].shift(), '')
        table = df_k.set_index(['gradient_type', 'timestep'])[['mae', 'spearman']]
        table_formatted = table.applymap(lambda x: f"{x:.3f}")
        print(f"\n=== Aggregated Table for kernel={k} ===")
        print(table_formatted)
        table_formatted.to_csv(f"agg_mae_spearman_kernel{k}.csv")

aggregate_tables(
    "sana_outputs/results_mae_post_layer_norm_latents.csv",
    "sana_outputs/results_spearman_post_layer_norm_latents.csv",
    kernels=(1, 3)
)

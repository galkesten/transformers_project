import pandas as pd

def aggregate_by_layer_component(mae_csv, spearman_csv, kernels=(1, 3)):
    df_mae = pd.read_csv(mae_csv)
    df_spearman = pd.read_csv(spearman_csv)
    
    key_cols = ['timestep', 'gradient_type', 'kernel', 'layer', 'component']
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
        # Optionally, sort for nice viewing
        df_k = df_k.sort_values(['gradient_type', 'layer', 'component'], ascending=[True, True, True])
        # Remove duplicate gradient_type for pretty printing, as before
        df_k['gradient_type'] = df_k['gradient_type'].mask(df_k['gradient_type'] == df_k['gradient_type'].shift(), '')
        table = df_k.set_index(['gradient_type', 'layer', 'component'])[['mae', 'spearman']]
        table_formatted = table.applymap(lambda x: f"{x:.3f}")
        print(f"\n=== Aggregated Table for kernel={k} (by layer+component) ===")
        print(table_formatted)
        table_formatted.to_csv(f"agg_mae_spearman_kernel{k}_by_layer_component.csv")

# Usage
aggregate_by_layer_component(
    "sana_outputs/results_mae_activations.csv",
    "sana_outputs/results_spearman_activations.csv",
    kernels=(1, 3)
)

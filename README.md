# SANA Transformer – Experiments

This repository contains three experiments that investigate the internal behavior of the **SANA text-to-image Transformer** by analyzing and modifying activations during the diffusion process.

---

## Experiment 1: Transformer Component Visualization
This experiment records and visualizes the **input and output activations** of the following components at **every timestep and block** in the transformer:
- Self-attention (`attn1`)
- Cross-attention (`attn2`)
- Feed-forward network (`ffn`)
For each denoising step, the activations from all 20 blocks are captured, reshaped, decoded using the VAE, and saved as images.
### Output Dir
exp1-sana_latent_vis_<prompt_slug>/
Visualization helper: after running 'exp1.ipynb`, run 'res_exp1_visualization.ipynb` to generate a figure showing the full activation chain (attn1 → attn2 → ff) for a single timestep.

---

## Experiment 2: FFN Zero and Mean Ablation
This experiment tests how the **FFN outputs** in the transformer affect generation by **replacing** them with:
- `zero`: all-zero tensors
- `mean`: the spatial or token-wise mean of the activation
Ablation is done in two modes:
#### Step-Wise
At each timestep `j`, all 20 FFNs are ablated.
####  Block-Wise
For each block `i`, its FFN is ablated across all timesteps.

###  Output Dir
exp2-sana_ablation_results_sana/
Visualization helper: after running 'exp2-part2.ipnb`, run ' res_exp2.ipynb` to view image grids comparing all ablations.

---

## Experiment 3: Component-Wise Spatial Transformation (Phase D)
This experiment applies **spatial transformations** to the **output activations** of a chosen component (e.g. `attn1`, `attn2`, or `ffn`) during generation.
The activation is intercepted, reshaped into `(B, C, H, W)`, transformed, and returned to the model.

### Transform Options
Set in the script using `transform_choice`. Supported values:
- `first_token`, `first_row`, `first_col`, `first_n_tokens`
- `swap_tb` – swap top and bottom halves
- `swap_lr` – swap left and right halves
- `swap_quadrants` – rotate corners
- `shuffle` – randomly permute spatial tokens
- `constant` – fill with flat mean

### Output Dir
results/





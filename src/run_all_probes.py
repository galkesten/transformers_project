import os
import argparse
import re

def find_timesteps_layers_components(folder):
    """
    Returns:
        dict of {timestep: {layer: [component,...] or [None]}}
    """
    result = {}
    items = os.listdir(folder)
    timestep_folders = [f for f in items if re.match(r'timestep_\d+', f) and os.path.isdir(os.path.join(folder, f))]
    if timestep_folders:
        # Folder-based structure
        for ts_folder in timestep_folders:
            ts_match = re.match(r'timestep_(\d+)', ts_folder)
            if not ts_match:
                continue
            ts = int(ts_match.group(1))
            result[ts] = {}
            ts_path = os.path.join(folder, ts_folder)
            for layer in os.listdir(ts_path):
                layer_path = os.path.join(ts_path, layer)
                if not os.path.isdir(layer_path):
                    continue
                comps = []
                for file in os.listdir(layer_path):
                    comp_match = re.match(r'([a-zA-Z0-9_]+)_accumulate_\d+\.pt$', file)
                    if comp_match:
                        comps.append(comp_match.group(1))
                if comps:
                    result[ts][layer] = comps
    else:
        # Flat, file-based structure
        for f in items:
            m = re.match(r'timestep_(\d+)_accumulate_\d+\.pt$', f)
            if m:
                ts = int(m.group(1))
                if ts not in result:
                    result[ts] = {None: [None]}
    return result

def run_probe_for_timestep(train_folder, test_folder, ts, args, layer=None, component=None, normalize_latents_with_layer_norm=False):
    script_name = "py-sbatch.sh"

    model_out = os.path.join(args.base_folder, "probes")
    test_out = os.path.join(args.base_folder, "evals")
    mae_csv = os.path.join(args.base_folder, f"results_mae_{args.latents_name}.csv")
    spearman_csv = os.path.join(args.base_folder, f"results_spearman_{args.latents_name}.csv")

    os.makedirs(model_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    layer_arg = f"--layer {layer}" if layer and layer != "None" else ""
    component_arg = f"--component {component}" if component and component != "None" else ""
    normalize_latents_with_layer_norm_arg = "--normalize_latents_with_layer_norm" if normalize_latents_with_layer_norm else ""

    cmd = f"""./{script_name} src/probe.py \
    --latents_folder_train {train_folder} \
    --latents_folder_test {test_folder} \
    --timestep {ts} \
    --accumulate_mode \
    --latent_type guided \
    --kernel_size {args.kernel_size} \
    --batch_size 16 \
    --models_output_folder {model_out} \
    --test_output_folder {test_out} \
    --test_results_file_path_mae {mae_csv} \
    --test_results_file_path_spearman {spearman_csv} \
    --gradient_type {args.gradient_type} \
    --accumulate_size 500 \
    {layer_arg} {component_arg} {normalize_latents_with_layer_norm_arg}
    """

    print(f"[Timestep {ts}] [Layer {layer}] [Component {component}] Submitting job with command:\n{cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--gradient_type", type=str, default="Vertical", choices=["Vertical", "Horizontal", "Gaussian"])
    parser.add_argument("--latents_name", type=str, required=True)
    parser.add_argument("--normalize_latents_with_layer_norm", action="store_true")
    args = parser.parse_args()

    train_folder = os.path.join(args.base_folder, "train", args.latents_name)
    test_folder = os.path.join(args.base_folder, "test",  args.latents_name)

    train_map = find_timesteps_layers_components(train_folder)
    test_map = find_timesteps_layers_components(test_folder)
    timesteps = sorted(set(train_map.keys()) & set(test_map.keys()))
    if not timesteps:
        raise ValueError("No matching timesteps found between train and test.")

    for ts in timesteps:
        train_layers = set(train_map[ts].keys())
        test_layers = set(test_map[ts].keys())
        for layer in (train_layers & test_layers):
            train_comps = set(train_map[ts][layer])
            test_comps = set(test_map[ts][layer])
            for component in (train_comps & test_comps):
                run_probe_for_timestep(train_folder, test_folder, ts, args, layer=layer, component=component, normalize_latents_with_layer_norm=args.normalize_latents_with_layer_norm)

if __name__ == "__main__":
    main()

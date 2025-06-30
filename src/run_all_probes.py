import os
import argparse

def find_timesteps(folder):
    files = os.listdir(folder)
    ts_set = set()
    for f in files:
        if f.startswith("timestep_") and "accumulate_" in f and f.endswith(".pt"):
            parts = f.split("_")
            # This assumes file format: timestep_{ts}_accumulate_{n}.pt
            ts_set.add(int(parts[1]))
    print(ts_set)
    return sorted(ts_set)

def run_probe_for_timestep(train_folder, test_folder, ts, args):
    script_name = "py-sbatch.sh"

    model_out = os.path.join(args.base_folder, "probes")
    test_out = os.path.join(args.base_folder, "evals")
    mae_csv = os.path.join(args.base_folder, f"results_mae_{args.latents_name}.csv")
    spearman_csv = os.path.join(args.base_folder, f"results_spearman_{args.latents_name}.csv")

    os.makedirs(model_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

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
    --accumulate_size 500
    """

    print(f"[Timestep {ts}] Submitting job with command:\n{cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--gradient_type", type=str, default="Vertical", choices=["Vertical", "Horizontal", "Gaussian"])
    parser.add_argument("--latents_name", type=str, required=True)
    args = parser.parse_args()

    train_folder = os.path.join(args.base_folder, "train", args.latents_name)
    test_folder = os.path.join(args.base_folder, "test",  args.latents_name)

    timesteps = sorted(list(set(find_timesteps(train_folder)) & set(find_timesteps(test_folder))))
    if not timesteps:
        raise ValueError("No matching timesteps found between train and test.")

    for ts in timesteps:
        run_probe_for_timestep(train_folder, test_folder, ts, args)

if __name__ == "__main__":
    main()

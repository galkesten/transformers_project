import os
import argparse

def find_timesteps(folder):
    files = os.listdir(folder)
    return sorted([
        int(f.split("_")[1].split(".")[0])
        for f in files if f.startswith("timestep_") and f.endswith(".pt")
    ])

def run_probe_for_timestep(train_path, test_path, ts, args):
    script_name = "py-sbatch.sh"

    model_out = os.path.join(args.base_folder, "probes")
    test_out = os.path.join(args.base_folder, "evals")
    mae_csv = os.path.join(args.base_folder, "mae_results.csv")
    spearman_csv = os.path.join(args.base_folder, "spearman_results.csv")

    os.makedirs(model_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    cmd = f"""./{script_name} src/probe.py \
    --train_path {train_path} \
    --test_path {test_path} \
    --kernel_size {args.kernel_size} \
    --batch_size 16 \
    --models_output_folder {model_out} \
    --test_output_folder {test_out} \
    --test_results_file_path_mae {mae_csv} \
    --test_results_file_path_spearman {spearman_csv} \
    --gradient_type {args.gradient_type}"""

    print(f"[Timestep {ts}] Submitting job with command:\n{cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--gradient_type", type=str, default="Vertical", choices=["Vertical", "Horizontal", "Gaussian"])
    args = parser.parse_args()

    train_dir = os.path.join(args.base_folder, "train", "latents")
    test_dir = os.path.join(args.base_folder, "test", "latents")

    timesteps = sorted(list(set(find_timesteps(train_dir)) & set(find_timesteps(test_dir))))
    if not timesteps:
        raise ValueError("No matching timesteps found between train and test.")

    for ts in timesteps:
        train_path = os.path.join(train_dir, f"timestep_{ts}.pt")
        test_path = os.path.join(test_dir, f"timestep_{ts}.pt")
        run_probe_for_timestep(train_path, test_path, ts, args)

if __name__ == "__main__":
    main()

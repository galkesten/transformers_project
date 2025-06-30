import torch
import os

def generate_random_latent_dataset_accumulate(
    output_path, num_samples=100, C=32, H=32, W=32,
    timestep=0, filename_prefix="baseline_timestep", accumulate_size=500
):
    os.makedirs(output_path, exist_ok=True)
    dataset = []
    num_accumulates = (num_samples + accumulate_size - 1) // accumulate_size

    for i in range(num_samples):
        latents = torch.randn(C, H, W)
        dataset.append({'latents': latents})
        # When reach accumulate_size or last sample, save and reset buffer
        if (i + 1) % accumulate_size == 0 or (i + 1) == num_samples:
            N = (i // accumulate_size) + 1
            filename = f"{filename_prefix}_{timestep}_accumulate_{N}.pt"
            torch.save(dataset, os.path.join(output_path, filename))
            print(f"Saved: {filename} [{len(dataset)} samples]")
            dataset = []

# Example usage
generate_random_latent_dataset_accumulate(
    output_path="sana_outputs/train/latents1",
    num_samples=10_000,
    C=2240,
    H=32,
    W=32,
    timestep=0,
    filename_prefix="train_baseline_timestep",
    accumulate_size=500
)

generate_random_latent_dataset_accumulate(
    output_path="sana_outputs/test/latents1",
    num_samples=1000,
    C=2240,
    H=32,
    W=32,
    timestep=0,
    filename_prefix="test_baseline_timestep",
    accumulate_size=500
)

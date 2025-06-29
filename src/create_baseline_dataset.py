import torch
import os

def generate_random_latent_dataset(output_path, num_samples=100, C=32, H=32, W=32, timestep=0, filename_prefix="baseline_timestep"):
    dataset = []
    for _ in range(num_samples):
        latents = torch.randn(C, H, W)  # Gaussian noise with mean=0, std=1
        dataset.append({'latents': latents})
    
    # Save with the expected naming convention
    filename = f"{filename_prefix}_{timestep}.pt"
    torch.save(dataset, os.path.join(output_path, filename))
    print(f"Saved: {filename}")

# Example usage
generate_random_latent_dataset(
    output_path="sana_outputs/train/latents",
    num_samples=10_000,
    C=32,
    H=32,
    W=32,
    timestep=0,
    filename_prefix="train_baseline_timestep")

generate_random_latent_dataset(
    output_path="sana_outputs/test/latents",
    num_samples=1000,
    C=32,
    H=32,
    W=32,
    timestep=0,
    filename_prefix="test_baseline_timestep")

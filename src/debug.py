import torch

if __name__ == "__main__":
    file_path = "sana_outputs/test/activations/timestep_999/layer_19/cross_attn_accumulate_1.pt"
    data = torch.load(file_path, map_location="cuda")

    for i, entry in enumerate(data):
        # Should be either entry['guided'], entry['unguided'], or entry['latents']
        for key in ['guided', 'unguided', 'latents']:
            if key in entry:
                tensor = entry[key]
                print(
                    f"Entry {i} | Key: {key} | shape: {tensor.shape} | min: {tensor.min().item():.3g} | "
                    f"max: {tensor.max().item():.3g} | mean: {tensor.mean().item():.3g} | std: {tensor.std().item():.3g} | "
                    f"nan: {torch.isnan(tensor).any().item()} | inf: {torch.isinf(tensor).any().item()}"
                )

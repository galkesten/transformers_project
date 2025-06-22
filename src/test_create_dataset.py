import torch
import os
import argparse

def inspect_latents_file(file_path):
    print(f"\n== Inspecting latents file: {file_path} ==")
    data = torch.load(file_path)
    for i, entry in enumerate(data):
        print(f"  Sample {i}:")
        print(f"    Prompt: {entry['prompt']}")
        print(f"    Seed: {entry['seed']}")
        print(f"    Step: {entry['step']}, Timestep: {entry['timestep']}")
        print(f"    Latents shape: {entry['latents'].shape}")
        
def inspect_activation_file(file_path):
    print(f"\n== Inspecting activation file: {file_path} ==")
    records = torch.load(file_path)
    print(f"  Number of records: {len(records)}")
    for i, r in enumerate(records[:3]):
        print(f"  Record {i}:")
        print(f"    Prompt: {r['prompt']}")
        print(f"    Seed: {r['seed']}")
        print(f"    Guided shape: {r['guided'].shape}")
        print(f"    Unguided shape: {r['unguided'].shape}")
        print(r['guided'])
        print(r['unguided'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--timestep", type=int, help="Timestep to inspect activations from")
    parser.add_argument("--layer", type=int, help="Layer to inspect activations from")
    parser.add_argument("--component", type=str, help="Component type (self_attn, cross_attn, mix_ffn)")
    parser.add_argument("--latents_file", type=str, help="Path to specific latents .pt file")
    args = parser.parse_args()

    if args.latents_file:
        inspect_latents_file(args.latents_file)

    if args.timestep is not None and args.layer is not None and args.component:
        act_path = os.path.join(
            args.output_dir,
            args.split,
            "activations",
            f"timestep_{args.timestep:03d}",
            f"layer_{args.layer:02d}",
            f"{args.component}.pt"
        )
        if os.path.exists(act_path):
            inspect_activation_file(act_path)
        else:
            print(f"Activation file not found: {act_path}")

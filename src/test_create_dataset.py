import torch
import os
import argparse

#change it to float 16 


def inspect_latents_file(file_path):
    print(f"\n== Inspecting latents file: {file_path} ==")
    data = torch.load(file_path)
    print(len(data))
    for i, entry in enumerate(data):
        if i>=10:
            break
        print(f"  Sample {i}:")
        print(f"    Prompt: {entry['prompt']}")
        print(f"    Seed: {entry['seed']}")
        print(f"    Step: {entry['step']}, Timestep: {entry['timestep']}")
        print(f"    Latents shape: {entry['latents'].shape}")


def inspect_post_layer_norms_latents_file(file_path):
    print(f"\n== Inspecting latents file: {file_path} ==")
    data = torch.load(file_path)
    print(len(data))
    for i, entry in enumerate(data):
        if i>=10:
            break
        print(f"  Sample {i}:")
        print(f"    Prompt: {entry['prompt']}")
        print(f"    Seed: {entry['seed']}")
        print(f"    Guided shape: {entry['guided'].shape}")
        print(f"    Unguided shape: {entry['unguided'].shape}")
        #print(entry['guided'])
        #print(entry['unguided'])
        
def inspect_activation_file(file_path):
    print(f"\n== Inspecting activation file: {file_path} ==")
    
    records = torch.load(file_path)
    print(len(records))
    print(f"  Number of records: {len(records)}")
    for i, r in enumerate(records[:3]):
        if i>=10:
            break
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
    parser.add_argument("--latents_file", type=str, help="Path to specific latents .pt file")
    parser.add_argument("--post_layernorm_latents_file", type=str, help="Path to specific latents .pt file")
    parser.add_argument("--activations_file", type=str, help="Path to specific latents .pt file")
    args = parser.parse_args()

    if args.latents_file:
        inspect_latents_file(args.latents_file)

    if args.post_layernorm_latents_file:
        inspect_post_layer_norms_latents_file(args.post_layernorm_latents_file)

    if args.activations_file:
        inspect_activation_file(args.activations_file)
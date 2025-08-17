"""Adapted from TODO"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import SanaPipeline
import sys


print(f"PYTHON EXECUTABLE: {sys.executable}", flush=True)
print(f"CONDA ENV: {os.environ.get('CONDA_DEFAULT_ENV')}", flush=True)

torch.set_grad_enabled(False)
N_STEPS    = 20

step_counter = -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )

    parser.add_argument(
        "--ablation_type",
        choices=["zero", "mean_per_token", "mean_over_tokens", "none"],
        default="none",
        help="type of ablation to apply",
    )
    parser.add_argument(
        "--ablation_layer",
        type=int,
        default=0,
        help="layer to apply ablation to",
    )
    parser.add_argument(
        "--ablation_component",
        choices=["self_attn", "cross_attn", "mix_ffn"],
        default="mix_ffn",
        help="component to apply ablation to",
    )
    parser.add_argument(
        "--mean_activations_file",
        type=str,
        default=None,
        help="file to load mean activations from",
    )
    #add list of timesteps
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        help="list of timesteps to apply ablation to",
    )
    #step wise
    parser.add_argument(
        "--step_wise",
        action="store_true",
        help="apply ablation step by step",
    )

    opt = parser.parse_args()
    return opt

def create_timestep_ablation_hook(timesteps):
    def zero_ablation_hook(module, input, output):
        global step_counter
        if step_counter not in timesteps:
            return output
        return torch.zeros_like(output)
    return zero_ablation_hook

def create_mean_per_token_ablation_hook(mean_activations, layer, timesteps):
    def mean_per_token_ablation_hook(module, input, output):
        global step_counter

        if step_counter not in timesteps:
            return output
        ret = mean_activations[layer][step_counter]
        #extend ret to the same shape as output
        ret = ret.expand(output.shape)
        #convert to dtype of output
        ret = ret.to(output.dtype)
        return ret
    return mean_per_token_ablation_hook

def create_mean_over_tokens_ablation_hook(timesteps):
    def mean_over_tokens_ablation_hook(module, input, output):
        global step_counter
        if step_counter not in timesteps:
            return output
        mean = output.mean(dim=1, keepdim=True) if output.ndim == 3 else output.mean(dim=(2,3), keepdim=True)
        return mean.expand_as(output)
    return mean_over_tokens_ablation_hook


def _count_steps(module, input):
    """Executed *before* each denoising step; increments global t."""
    global step_counter 
    step_counter = step_counter + 1 

def load_model():
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    return pipe

def register_component_hooks_per_layer(model, block_id, component_type, ablation_type, mean_activations=None, timesteps=None):
    handles = []
    transformer_blocks = model.transformer_blocks
    ablation_hook = None
    block = transformer_blocks[block_id]
    print(block)
    print(f"ablation_type: {ablation_type}")
    print(f"component_type: {component_type}")
    if mean_activations is not None:
        print(f"mean_activations: {mean_activations.shape}")

    if ablation_type == "zero":
        ablation_hook = create_timestep_ablation_hook(timesteps)
        if component_type == "self_attn":
            handles.append(block.attn1.register_forward_hook(ablation_hook))
        if component_type == "cross_attn":
            handles.append(block.attn2.register_forward_hook(ablation_hook))
        if component_type == "mix_ffn":
            handles.append(block.ff.register_forward_hook(ablation_hook))  
    elif ablation_type == "mean_per_token":
        ablation_hook = create_mean_per_token_ablation_hook(mean_activations, block_id, timesteps)
        if component_type == "self_attn":
            handles.append(block.attn1.register_forward_hook(ablation_hook))
        if component_type == "cross_attn":
            handles.append(block.attn2.register_forward_hook(ablation_hook))
        if component_type == "mix_ffn":
            handles.append(block.ff.register_forward_hook(ablation_hook))
    elif ablation_type == "mean_over_tokens":
        ablation_hook = create_mean_over_tokens_ablation_hook(timesteps)
        if component_type == "self_attn":
            handles.append(block.attn1.register_forward_hook(ablation_hook))
        if component_type == "cross_attn":
            handles.append(block.attn2.register_forward_hook(ablation_hook))
        if component_type == "mix_ffn":
            handles.append(block.ff.register_forward_hook(ablation_hook))
    return handles

def register_component_hooks_per_step(model, component_type, ablation_type, mean_activations=None, timesteps=None):
    handles = []
    transformer_blocks = model.transformer_blocks
    for i, block in enumerate(transformer_blocks):
        if ablation_type == "zero":
            ablation_hook = create_timestep_ablation_hook(timesteps)
            if component_type == "self_attn":
                handles.append(block.attn1.register_forward_hook(ablation_hook))
            if component_type == "cross_attn":
                handles.append(block.attn2.register_forward_hook(ablation_hook))
            if component_type == "mix_ffn":
                handles.append(block.ff.register_forward_hook(ablation_hook))  
        elif ablation_type == "mean_per_token":
            ablation_hook = create_mean_per_token_ablation_hook(mean_activations, i, timesteps)
            if component_type == "self_attn":
                handles.append(block.attn1.register_forward_hook(ablation_hook))
            if component_type == "cross_attn":
                handles.append(block.attn2.register_forward_hook(ablation_hook))
            if component_type == "mix_ffn":
                handles.append(block.ff.register_forward_hook(ablation_hook))
        elif ablation_type == "mean_over_tokens":
            ablation_hook = create_mean_over_tokens_ablation_hook(timesteps)
            if component_type == "self_attn":
                handles.append(block.attn1.register_forward_hook(ablation_hook))
            if component_type == "cross_attn":
                handles.append(block.attn2.register_forward_hook(ablation_hook))
            if component_type == "mix_ffn":
                handles.append(block.ff.register_forward_hook(ablation_hook))
    return handles

def main(opt):
    if opt.ablation_type == "mean_per_token":
        mean_activations = torch.load(opt.mean_activations_file)
        print(f"mean_activations: {mean_activations.shape}")
    else:
        mean_activations = None
    print(f"timesteps: {opt.timesteps}")
    print(f"step_wise: {opt.step_wise}")
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model
    model = load_model()
    model.transformer.register_forward_pre_hook(_count_steps)

    global step_counter

    if opt.ablation_type != "none":
        print(f"Ablating {opt.ablation_type} {opt.ablation_component} in layer {opt.ablation_layer}") 
        if opt.step_wise:
            handles = register_component_hooks_per_step(model.transformer, opt.ablation_component, opt.ablation_type, mean_activations, opt.timesteps)
        else:
            handles = register_component_hooks_per_layer(model.transformer, opt.ablation_layer, opt.ablation_component, opt.ablation_type, mean_activations, opt.timesteps)

    for index, metadata in enumerate(metadatas):
        step_counter = -1 #reset step counter for each prompt
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images
                samples = model(
                    prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                ).images
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                print("creating grid")
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
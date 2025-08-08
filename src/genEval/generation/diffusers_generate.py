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
        choices=["zero", "none"],
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


    opt = parser.parse_args()
    return opt

def zero_ablation_hook(module, input, output):
    return torch.zeros_like(output)


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

def register_component_hooks(model, block_id, component_type, ablation_type):
    handles = []
    transformer_blocks = model.transformer_blocks
    ablation_hook = None
    if ablation_type == "zero":
        ablation_hook = zero_ablation_hook

    block = transformer_blocks[block_id]
    print(block)
    if component_type == "self_attn":
         handles.append(block.attn1.register_forward_hook(ablation_hook))
    if component_type == "cross_attn":
        handles.append(block.attn2.register_forward_hook(ablation_hook))
    if component_type == "mix_ffn":
        handles.append(block.ff.register_forward_hook(ablation_hook))  
    return handles

def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model
    model = load_model()
    if opt.ablation_type != "none":
        print(f"Ablating {opt.ablation_type} {opt.ablation_component} in layer {opt.ablation_layer}")   
        handles = register_component_hooks(model.transformer, opt.ablation_layer, opt.ablation_component, opt.ablation_type)

    for index, metadata in enumerate(metadatas):
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
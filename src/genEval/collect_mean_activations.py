import os
import torch, tqdm 
from diffusers import SanaPipeline
import re, matplotlib.pyplot as plt
from PIL import Image
import json
from pytorch_lightning import seed_everything
import argparse
from tqdm import tqdm, trange

step_counter = -1
torch.set_grad_enabled(False)
N_STEPS    = 20
LEVELS = None
DEVICE = "cuda"
means = None

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
        "--ablation_component",
        choices=["self_attn", "cross_attn", "mix_ffn"],
        default="mix_ffn",
        help="component to apply ablation to",
    )

    opt = parser.parse_args()
    return opt
    
def load_model():
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe.to(DEVICE)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    return pipe

def register_component_hooks(model, component_type):
    handles = []
    transformer_blocks = model.transformer_blocks
    for i, block in enumerate(transformer_blocks):
        if component_type == "self_attn":
            handles.append(block.attn1.register_forward_hook(make_mean_hook(i)))
        if component_type == "cross_attn":
            handles.append(block.attn2.register_forward_hook(make_mean_hook(i)))
        if component_type == "mix_ffn":
            handles.append(block.ff.register_forward_hook(make_mean_hook(i)))
    return handles

def load_prompt_file(json_prompt_file):
    prompts = []
    with open(json_prompt_file, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    return prompts


def _count_steps(module, input):
    """Executed *before* each denoising step; increments global t."""
    global step_counter 
    step_counter = step_counter + 1 

def make_mean_hook(layer:int):

    def collect_mean_activations_hook(module, input, output):
        global means
        global step_counter
        print(f"step_counter: {step_counter}")
        print(f"output: {output.shape}")
        #sum over batch dimension
        copied_output = output.clone()
        copied_output = copied_output.sum(dim=0)
        print(f"copied_output: {copied_output.shape}")
        if means[layer][step_counter] is None:
            means[layer][step_counter] = copied_output
        else:
            means[layer][step_counter] += copied_output

        return output
    
    return collect_mean_activations_hook


if __name__ == "__main__":
    opt = parse_args()
    pipe = load_model()
    pipe.transformer.register_forward_pre_hook(_count_steps)
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    LEVELS = len(pipe.transformer.transformer_blocks)
    means = [[None] * N_STEPS for _ in range(LEVELS)]
    handles = register_component_hooks(pipe.transformer, opt.ablation_component)

    # Create output directory once
    os.makedirs(opt.outdir, exist_ok=True)
    
    total_samples = 0
    for index, metadata in enumerate(metadatas):
        step_counter = -1 #reset step counter for each prompt
        seed_everything(opt.seed)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_count = 0

        with torch.no_grad():
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images
                samples = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=N_STEPS,
                    guidance_scale=5.0,
                    num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                ).images
                print(f"len(samples): {len(samples)}")
                for sample in samples:
                    sample_count += 1
                total_samples += len(samples)

    dominator = total_samples * 2 # classifier free guidance use 2 samples per prompt
   # 1. Replace None with zeros and divide by dominator
    for i in range(LEVELS):
        for j in range(N_STEPS):
            if means[i][j] is None:
                means[i][j] = torch.zeros_like(means[0][0])
            means[i][j] = means[i][j] / dominator

    # 2. Stack to tensor
    means_tensor = torch.stack([torch.stack(row, dim=0) for row in means], dim=0)
   
    # 3. Save
    torch.save(means_tensor, os.path.join(opt.outdir, f"mean_activations_{opt.ablation_component}.pt"))

    print(total_samples)








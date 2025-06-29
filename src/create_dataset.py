import torch
import os
import json
import random
import argparse
from collections import defaultdict
from diffusers import SanaPipeline
from huggingface_hub import hf_hub_download
from diffusers.pipelines.sana.pipeline_sana import retrieve_timesteps

current_timestep = None
all_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
latents_by_ts = defaultdict(list)
post_final_layer_norm_latents_by_ts = defaultdict(list)

def reset_globals():
    global current_timestep, all_activations, latents_by_ts, post_final_layer_norm_latents_by_ts
    current_timestep = None
    all_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    latents_by_ts = defaultdict(list)
    post_final_layer_norm_latents_by_ts = defaultdict(list)

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

def create_callback(latents_by_ts, prompts, seed, timesteps):
    def debug_callback_on_step_end(pipeline, step, timestep, callback_kwargs):
        global current_timestep
       
        latents = callback_kwargs["latents"]

        if timestep not in timesteps:
            return {"latents": latents}
        
        current_timestep = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        for i, prompt in enumerate(prompts):
             latents_by_ts[current_timestep].append({
                "step": step,
                "timestep": current_timestep,
                "latents": latents[i].detach().cpu(),
                "prompt": prompt,
                "seed": seed,
            })
             
        return {"latents": latents}
    
    return debug_callback_on_step_end

def make_component_hook(component_type, layer_id, prompts, seed, timesteps):
    def hook_fn(mod, inp, out):
        global current_timestep, all_activations, latents_by_ts
        if current_timestep not in timesteps:
            return
        
        if not isinstance(out, torch.Tensor):
            raise ValueError("Unexpected Output")
        if out.dim() == 3:
            b, l, d = out.shape
            shaped_out = out.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
        elif out.dim() == 4:
            shaped_out = out
        else:
            raise ValueError(f"Unexpected tensor shape: {out.shape}")

        mid = shaped_out.shape[0] // 2
        unguided = shaped_out[:mid].detach().cpu()
        guided = shaped_out[mid:].detach().cpu()

        for i in range(len(prompts)):
            all_activations[current_timestep][layer_id][component_type].append({
                "prompt": prompts[i],
                "current_timestep": current_timestep,
                "seed": seed,
                "guided": guided[i],
                "unguided": unguided[i],         
            })
    return hook_fn

def time_hook(mod, inp, out):
    global current_timestep
    current_timestep = int(inp[0][0].item())

def register_time_step_hook(model):
    handles = []
    handles.append(model.time_embed.register_forward_hook(time_hook))
    return handles

def register_component_hooks(model, prompts, seed, layers, timesteps, component_type):
    handles = []
    transformer_blocks = model.transformer_blocks
    for i, block in enumerate(transformer_blocks):
        if i not in layers:
            continue
        if component_type == "self_attn":
            handles.append(block.attn1.register_forward_hook(make_component_hook("self_attn", i, prompts, seed, timesteps)))
        elif component_type == "cross_attn":
            handles.append(block.attn2.register_forward_hook(make_component_hook("cross_attn", i, prompts, seed, timesteps)))
        elif component_type == "mix_ffn":
            handles.append(block.ff.register_forward_hook(make_component_hook("mix_ffn", i, prompts, seed, timesteps)))
    return handles

def make_post_final_layer_norm_latents_hook(prompts, seed, timesteps):
    def hook_fn(mod, inp, out):
        global current_timestep, post_final_layer_norm_latents_by_ts
        if current_timestep not in timesteps:
            return
        if not isinstance(inp[0], torch.Tensor):
            raise ValueError("Unexpected Output")
        if inp[0].dim() == 3:
            b, l, d =inp[0].shape
            shaped_in = inp[0].unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
        elif inp[0].dim() == 4:
            shaped_in = inp[0]
        else:
            raise ValueError(f"Unexpected tensor shape: {inp.shape}")

        mid = shaped_in.shape[0] // 2
        unguided = shaped_in[:mid].detach().cpu()
        guided = shaped_in[mid:].detach().cpu()

        for i in range(len(prompts)):
            post_final_layer_norm_latents_by_ts[current_timestep].append({
                "prompt": prompts[i],
                "current_timestep": current_timestep,
                "seed": seed,
                "guided": guided[i],
                "unguided": unguided[i],         
            })
    return hook_fn

def register_post_final_layer_norm_latents(model, prompts, seed, timesteps):
    handles = []
    handles.append(model.proj_out.register_forward_hook(make_post_final_layer_norm_latents_hook(prompts, seed, timesteps)))
    return handles

def sample_prompts(n_train, n_test, seed=42):
    json_path = hf_hub_download(
        repo_id="playgroundai/MJHQ-30K",
        filename="meta_data.json",
        repo_type="dataset"
    )
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts = [info["prompt"] for info in data.values()]
    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:n_train], prompts[n_train:n_train + n_test]

def get_timesteps(pipe, timesteps_step_size=5):
    timesteps = retrieve_timesteps(pipe.scheduler, 20)[0]
    if isinstance(timesteps, list):
        timesteps = torch.tensor([int(t) if torch.is_tensor(t) else t for t in timesteps])

    results = []

    for i, t in enumerate(timesteps):
        if i % timesteps_step_size == 0:
            results.append(t)
        elif i == len(timesteps) - 1:
            results.append(t)

    return results
    
def get_layers(pipe, layers_step_size=5):
    transformer_blocks = pipe.transformer_blocks
    num_layers = len(transformer_blocks)
    results = list(range(0, num_layers, layers_step_size))

    if (num_layers - 1) not in results:
        results.append(num_layers - 1)

    return results


def generate_activations(prompts, pipe, save_activations, save_latents, save_post_final_layer_norm_latents, layers, timesteps, component_type):
    seed = random.randint(0, 99999)
    print(f"[INFO] Generating activations for {len(prompts)} prompt(s) | Seed: {seed}")
    hooks = []
    hooks += register_time_step_hook(pipe.transformer)
    if save_activations:
        hooks += register_component_hooks(pipe.transformer, prompts, seed, layers, timesteps, component_type)

    if save_post_final_layer_norm_latents:
        hooks += register_post_final_layer_norm_latents(pipe.transformer, prompts, seed, timesteps)

    callback_on_step_end=create_callback(latents_by_ts, prompts, seed, timesteps) if save_latents else None
    callback_on_step_end_tensor_inputs=["latents"] if save_latents else None
    _ = pipe(
        prompt=prompts,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        callback_on_step_end= callback_on_step_end,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
    )
    print(f"[INFO] Finished generation for {len(prompts)} prompt(s)")
    if save_activations or save_post_final_layer_norm_latents:
        for h in hooks:
            h.remove()

def save_outputs(output_base, split, save_activations, save_latents, save_post_final_layer_norm_latents, accumulate_counter):
    
    global all_activations, latents_by_ts
    if save_activations:
        print(f"[INFO] Saving activations for split: {split}")
        for timestep, layers in all_activations.items():
            for layer_id, components in layers.items():
                for component_type, records in components.items():
                    #print(f"timestep : {timestep}, layer id : {layer_id} component type: {component_type}")
                    out_path = os.path.join(output_base, split, "activations", f"timestep_{timestep:03d}", f"layer_{layer_id:02d}", f"{component_type}_accumulate_{accumulate_counter}.pt")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    torch.save(records, out_path)

        print(f"[INFO] Saved all activations for split: {split}")

    if save_latents:
        print(f"[INFO] Saving latents for split: {split}")
        for timestep, sample_dict in latents_by_ts.items():
            out_path = os.path.join(output_base, split, "latents", f"timestep_{timestep:03d}_accumulate_{accumulate_counter}.pt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(sample_dict, out_path)
        print(f"[INFO] Saved all latents for split: {split}")
    
    if save_post_final_layer_norm_latents:
        print(f"[INFO] Saving post final layer norm latents for split: {split}")
        for timestep, sample_dict in post_final_layer_norm_latents_by_ts.items():
            out_path = os.path.join(output_base, split, "post_layer_norm_latents", f"timestep_{timestep:03d}_accumulate_{accumulate_counter}.pt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(sample_dict, out_path)
        print(f"[INFO] Saved post layer norm latents for split: {split}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_latents", action="store_true", help="Whether to save latents.")
    parser.add_argument("--save_activations", action="store_true", help="Whether to save activations.")
    parser.add_argument("--save_post_final_layer_norm_latents", action="store_true", help="Whether to save post-final-layer-norm latents.")
    parser.add_argument("--layers_step", type=int, default=5, help="Layer step size")
    parser.add_argument("--timesteps_step", type=int, default=1)
    parser.add_argument("--component_type", type=str, choices=["self_attn", "cross_attn", "mix_ffn"], default="mix_ffn")

    args = parser.parse_args()

    print(f"args.save latents: {args.save_latents}")
    print(f"args.save activations: {args.save_activations}")
    print(f"args.save_post_final_layer_norm_latents: {args.save_post_final_layer_norm_latents}")
    print("[INFO] Sampling prompts...")

    train_prompts, test_prompts = sample_prompts(args.n_train, args.n_test, args.seed)

    pipe = load_model()

    timesteps = get_timesteps(pipe=pipe, timesteps_step_size=args.timesteps_step)
    layers = get_layers(pipe=pipe.transformer, layers_step_size=args.layers_step)
    
    accumulate = 0
    accumulate_counter = 0
    print("[INFO] Starting training generation...")
    for i in range(0, len(train_prompts), args.batch_size):
        print(f"[INFO] Processing training batch {i // args.batch_size + 1}/{(len(train_prompts) + args.batch_size - 1) // args.batch_size}")
        batch = train_prompts[i:i + args.batch_size]
        generate_activations(batch, pipe,  save_activations=args.save_activations, save_latents=args.save_latents,
                             save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents,
                             layers=layers,
                             timesteps=timesteps,
                             component_type=args.component_type)
        accumulate += len(batch)
        if accumulate >= 500:
            accumulate_counter = accumulate_counter + 1
            save_outputs(args.output_dir, split="train", save_activations=args.save_activations, 
                         save_latents=args.save_latents, save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents,
                         accumulate_counter=accumulate_counter)
            accumulate = 0
            reset_globals()

    if accumulate != 0:
        accumulate_counter = accumulate_counter + 1
        save_outputs(args.output_dir, split="train", save_activations=args.save_activations, 
                         save_latents=args.save_latents, save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents,
                         accumulate_counter=accumulate_counter)
    reset_globals()
    accumulate = 0
    accumulate_counter = 0

    print("[INFO] Starting testing generation...")
    for i in range(0, len(test_prompts), args.batch_size):
        batch = test_prompts[i:i + args.batch_size]
        print(f"[INFO] Processing testing batch {i // args.batch_size + 1}/{(len(test_prompts) + args.batch_size - 1) // args.batch_size}")
        generate_activations(batch, pipe, save_activations=args.save_activations, save_latents=args.save_latents, 
                             save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents, 
                             layers=layers,
                             timesteps=timesteps,
                             component_type=args.component_type)
        accumulate += len(batch)
        if accumulate >= 500:
            accumulate_counter = accumulate_counter + 1
            save_outputs(args.output_dir, split="test", save_activations=args.save_activations, 
                         save_latents=args.save_latents, save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents,
                         accumulate_counter=accumulate_counter)
            accumulate = 0
            reset_globals()
    
    if accumulate != 0:
        accumulate_counter = accumulate_counter + 1
        save_outputs(args.output_dir, split="test", save_activations=args.save_activations, 
                         save_latents=args.save_latents, save_post_final_layer_norm_latents=args.save_post_final_layer_norm_latents,
                         accumulate_counter=accumulate_counter)
    reset_globals()
    accumulate = 0
    accumulate_counter = 0
    print("[INFO] Done.")

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import csv
import random
import uuid
from scipy.stats import spearmanr
from collections import defaultdict
from diffusers import SanaPipeline
from huggingface_hub import hf_hub_download
from diffusers.pipelines.sana.pipeline_sana import retrieve_timesteps
import json
import types
import matplotlib.pyplot as plt
import torch.nn.functional as F

# === Fixed Training Hyperparameters ===
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4

# Global state
current_timestep = None
probes_dict = {}  # {(timestep, layer/position, component): probe}
optimizers_dict = {}  # {(timestep, layer/position, component): optimizer}
targets_dict = {}  # {(timestep, layer/position, component): target}
training_mode = True
normalize_latents_with_layer_norm = False
layer_norm = None
eval_metrics = defaultdict(lambda: defaultdict(list))  # {key: {'mae': [...], 'spearman': [...]}}

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

def generate_gradient_map(H, W, map_type):
    if map_type not in ["Horizontal", "Vertical", "Gaussian"]:
        raise ValueError(f"{map_type} not supported")

    if map_type == "Horizontal":
        gradient = torch.linspace(0, 1, steps=W).repeat(H, 1)
    elif map_type == "Vertical":
        gradient = torch.linspace(0, 1, steps=H).view(H, 1).repeat(1, W)
    elif map_type == "Gaussian":
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        xv, yv = np.meshgrid(x, y)
        sigma = 0.5
        gauss = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
        gradient = torch.tensor(gauss, dtype=torch.float32)

    return gradient

def initialize_probe(C, H, W, kernel_size, gradient_type, device):
    """Initialize a probe and its target"""
    probe = nn.Conv2d(C, 1, kernel_size=(kernel_size, kernel_size), stride=1, padding=0).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    
    # Create target
    with torch.no_grad():
        dummy_input = torch.zeros(1, C, H, W).to(device)
        output = probe(dummy_input)
        _, _, out_H, out_W = output.shape
        target = generate_gradient_map(out_H, out_W, gradient_type).unsqueeze(0).unsqueeze(0).to(device)
    
    return probe, target

def initialize_all_probes(timesteps, components_config, kernel_size, gradient_types, device, sample_shape):
    """Initialize all probes for all timestep/position/component/gradient_type combinations
    
    components_config is a dict: {component_name: [list of positions]}
    """
    global probes_dict, optimizers_dict, targets_dict
    
    C, H, W = sample_shape[1], sample_shape[2], sample_shape[3]
    
    for ts in timesteps:
        for component, comp_positions in components_config.items():
            for pos in comp_positions:
                for grad_type in gradient_types:
                    key = (ts, pos, component, grad_type)
                    probe, target = initialize_probe(C, H, W, kernel_size, grad_type, device)
                    optimizer = torch.optim.Adam(probe.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
                    
                    probes_dict[key] = probe
                    optimizers_dict[key] = optimizer
                    targets_dict[key] = target
                    
                    print(f"Initialized probe for timestep={ts}, position={pos}, component={component}, gradient={grad_type}")

def time_hook(mod, inp, out):
    global current_timestep
    current_timestep = int(inp[0][0].item())
    print(f"Current timestep: {current_timestep}")

def make_component_hook(position, component_type, timesteps, gradient_types):
    """Hook that performs backward pass for the relevant probe during training"""
    def hook_fn(mod, inp, out):
        global current_timestep, training_mode, normalize_latents_with_layer_norm, layer_norm
        
        if current_timestep not in timesteps:
            return
        
        if not isinstance(out, torch.Tensor):
            return
        
        # Reshape output to [B, C, H, W]
        if out.dim() == 3:
            b, l, d = out.shape
            shaped_out = out.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
        elif out.dim() == 4:
            shaped_out = out
        else:
            return
        
        # Extract guided activations (second half of batch)
        mid = shaped_out.shape[0] // 2
        guided = shaped_out[mid:].detach()  # Detach to prevent gradients flowing back to model
        
        # Normalize if needed (for both training and evaluation)
        if normalize_latents_with_layer_norm and layer_norm is not None:
            B, C, H, W = guided.shape
            guided = guided.permute(0, 2, 3, 1)
            with torch.no_grad():
                guided = layer_norm(guided)
            guided = guided.permute(0, 3, 1, 2)
        
        if training_mode:
            # Training: perform backward pass for each gradient type
            B = guided.shape[0]
            
            for grad_type in gradient_types:
                key = (current_timestep, position, component_type, grad_type)
                if key not in probes_dict:
                    raise ValueError(f"Probe not found for key: {key}")
                
                probe = probes_dict[key]
                optimizer = optimizers_dict[key]
                target = targets_dict[key]
                
                optimizer.zero_grad()
                output = probe(guided)
                loss = nn.MSELoss()(output, target.expand(B, -1, -1, -1))
                loss.backward()
                optimizer.step()
        else:
            # Evaluation: evaluate on the fly and accumulate metrics
            B = guided.shape[0]
            
            for grad_type in gradient_types:
                key = (current_timestep, position, component_type, grad_type)
                if key not in probes_dict:
                    raise ValueError(f"Probe not found for key: {key}")
                
                probe = probes_dict[key]
                target = targets_dict[key]
                
                probe.eval()
                with torch.no_grad():
                    output = probe(guided)
                    truth = target.expand(B, -1, -1, -1)
                    
                    # Compute metrics for each sample in batch
                    for j in range(B):
                        pred_flat = output[j].flatten().cpu().numpy()
                        target_flat = truth[j].flatten().cpu().numpy()
                        
                        # MAE
                        mae = np.abs(pred_flat - target_flat).mean()
                        eval_metrics[key]['mae'].append(mae)
                        
                        # Spearman correlation
                        corr, _ = spearmanr(pred_flat, target_flat)
                        if not np.isnan(corr):
                            eval_metrics[key]['spearman'].append(corr)
    
    return hook_fn

def register_time_step_hook(model):
    handles = []
    handles.append(model.time_embed.register_forward_hook(time_hook))
    return handles

def register_component_hooks(model, layers, timesteps, component_type, use_contributions_mode, gradient_types, train_on_block_output=False):
    handles = []
    transformer_blocks = model.transformer_blocks
    
    if train_on_block_output:
        # Train on entire block output
        for i, block in enumerate(transformer_blocks):
            if i not in layers:
                continue
            # Hook to the block's output (after all components)
            handles.append(block.register_forward_hook(
                make_component_hook(i, "block_output", timesteps, gradient_types)))
    else:
        # Train on individual components
        for i, block in enumerate(transformer_blocks):
            if i not in layers:
                continue
            if component_type == "self_attn" or component_type == "self_attn_after_gate":
                if use_contributions_mode or component_type == "self_attn_after_gate":
                    handles.append(block.identity_after_attn.register_forward_hook(
                        make_component_hook(i, "self_attn_after_gate", timesteps, gradient_types)))
                else:
                    handles.append(block.attn1.register_forward_hook(
                        make_component_hook(i, "self_attn", timesteps, gradient_types)))
            elif component_type == "cross_attn":
                handles.append(block.attn2.register_forward_hook(
                    make_component_hook(i, "cross_attn", timesteps, gradient_types)))
            elif component_type == "mix_ffn" or component_type == "mix_ffn_after_gate":
                if use_contributions_mode or component_type == "mix_ffn_after_gate":
                    handles.append(block.identity_after_ff.register_forward_hook(
                        make_component_hook(i, "mix_ffn_after_gate", timesteps, gradient_types)))
                else:
                    handles.append(block.ff.register_forward_hook(
                        make_component_hook(i, "mix_ffn", timesteps, gradient_types)))
    return handles

def register_patch_embed_hook(model, timesteps, gradient_types):
    """Register hook for patch_embed output (initial representation)"""
    def make_patch_embed_hook(timesteps, gradient_types):
        def hook_fn(mod, inp, out):
            global current_timestep, training_mode, normalize_latents_with_layer_norm, layer_norm
            
            if current_timestep not in timesteps:
                return
            
            if not isinstance(out, torch.Tensor):
                return
            
            # Reshape output to [B, C, H, W]
            if out.dim() == 3:
                b, l, d = out.shape
                shaped_out = out.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
            elif out.dim() == 4:
                shaped_out = out
            else:
                return
            
            # Extract guided activations
            mid = shaped_out.shape[0] // 2
            guided = shaped_out[mid:].detach()
            
            # Normalize if needed (for both training and evaluation)
            if normalize_latents_with_layer_norm and layer_norm is not None:
                B, C, H, W = guided.shape
                guided = guided.permute(0, 2, 3, 1)
                with torch.no_grad():
                    guided = layer_norm(guided)
                guided = guided.permute(0, 3, 1, 2)
            
            if training_mode:
                # Training: perform backward pass for each gradient type
                B = guided.shape[0]
                
                for grad_type in gradient_types:
                    key = (current_timestep, "initial", "patch_embed", grad_type)
                    if key not in probes_dict:
                        continue
                    
                    probe = probes_dict[key]
                    optimizer = optimizers_dict[key]
                    target = targets_dict[key]
                    
                    optimizer.zero_grad()
                    output = probe(guided)
                    loss = nn.MSELoss()(output, target.expand(B, -1, -1, -1))
                    loss.backward()
                    optimizer.step()
            else:
                # Evaluation: evaluate on the fly and accumulate metrics
                B = guided.shape[0]
                
                for grad_type in gradient_types:
                    key = (current_timestep, "initial", "patch_embed", grad_type)
                    if key not in probes_dict:
                        continue
                    
                    probe = probes_dict[key]
                    target = targets_dict[key]
                    
                    probe.eval()
                    with torch.no_grad():
                        output = probe(guided)
                        truth = target.expand(B, -1, -1, -1)
                        
                        # Compute metrics for each sample in batch
                        for j in range(B):
                            pred_flat = output[j].flatten().cpu().numpy()
                            target_flat = truth[j].flatten().cpu().numpy()
                            
                            # MAE
                            mae = np.abs(pred_flat - target_flat).mean()
                            eval_metrics[key]['mae'].append(mae)
                            
                            # Spearman correlation
                            corr, _ = spearmanr(pred_flat, target_flat)
                            if not np.isnan(corr):
                                eval_metrics[key]['spearman'].append(corr)
        
        return hook_fn
    
    handles = []
    handles.append(model.patch_embed.register_forward_hook(make_patch_embed_hook(timesteps, gradient_types)))
    return handles

def register_proj_out_hook(model, timesteps, gradient_types):
    """Register hook for proj_out output (final representation)"""
    def make_proj_out_hook(timesteps, gradient_types):
        def hook_fn(mod, inp, out):
            global current_timestep, training_mode, normalize_latents_with_layer_norm, layer_norm
            
            if current_timestep not in timesteps:
                return
            
            if not isinstance(inp[0], torch.Tensor):
                return
            
            # Reshape input to [B, C, H, W]
            inp_tensor = inp[0]
            if inp_tensor.dim() == 3:
                b, l, d = inp_tensor.shape
                shaped_inp = inp_tensor.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
            elif inp_tensor.dim() == 4:
                shaped_inp = inp_tensor
            else:
                return
            
            # Extract guided activations
            mid = shaped_inp.shape[0] // 2
            guided = shaped_inp[mid:].detach()
            
            # Normalize if needed (for both training and evaluation)
            if normalize_latents_with_layer_norm and layer_norm is not None:
                B, C, H, W = guided.shape
                guided = guided.permute(0, 2, 3, 1)
                with torch.no_grad():
                    guided = layer_norm(guided)
                guided = guided.permute(0, 3, 1, 2)
            
            if training_mode:
                # Training: perform backward pass for each gradient type
                B = guided.shape[0]
                
                for grad_type in gradient_types:
                    key = (current_timestep, "final", "proj_out", grad_type)
                    if key not in probes_dict:
                        continue
                    
                    probe = probes_dict[key]
                    optimizer = optimizers_dict[key]
                    target = targets_dict[key]
                    
                    optimizer.zero_grad()
                    output = probe(guided)
                    loss = nn.MSELoss()(output, target.expand(B, -1, -1, -1))
                    loss.backward()
                    optimizer.step()
            else:
                # Evaluation: evaluate on the fly and accumulate metrics
                B = guided.shape[0]
                
                for grad_type in gradient_types:
                    key = (current_timestep, "final", "proj_out", grad_type)
                    if key not in probes_dict:
                        continue
                    
                    probe = probes_dict[key]
                    target = targets_dict[key]
                    
                    probe.eval()
                    with torch.no_grad():
                        output = probe(guided)
                        truth = target.expand(B, -1, -1, -1)
                        
                        # Compute metrics for each sample in batch
                        for j in range(B):
                            pred_flat = output[j].flatten().cpu().numpy()
                            target_flat = truth[j].flatten().cpu().numpy()
                            
                            # MAE
                            mae = np.abs(pred_flat - target_flat).mean()
                            eval_metrics[key]['mae'].append(mae)
                            
                            # Spearman correlation
                            corr, _ = spearmanr(pred_flat, target_flat)
                            if not np.isnan(corr):
                                eval_metrics[key]['spearman'].append(corr)
        
        return hook_fn
    
    handles = []
    handles.append(model.proj_out.register_forward_hook(make_proj_out_hook(timesteps, gradient_types)))
    return handles

def install_forward_with_identities(block):
    block.identity_after_attn = nn.Identity()
    block.identity_after_ff = nn.Identity()

    def forward2(self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + self.identity_after_attn(gate_msa * attn_output)

        # Cross-Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)

        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + self.identity_after_ff(gate_mlp * ff_output)

        return hidden_states

    block.forward = types.MethodType(forward2, block)

def sample_prompts(n_train, n_test):
    """Sample prompts from dataset. Assumes random.seed() has been set by caller."""
    json_path = hf_hub_download(
        repo_id="playgroundai/MJHQ-30K",
        filename="meta_data.json",
        repo_type="dataset"
    )
    with open(json_path, 'r') as f:
        data = json.load(f)

    prompts = [info["prompt"] for info in data.values()]
    random.shuffle(prompts)  # Uses global random state
    return prompts[:n_train], prompts[n_train:n_train + n_test]

class PromptDataset(Dataset):
    """Simple dataset for prompts"""
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]

def get_all_timesteps(pipe):
    """Get all timesteps from the scheduler"""
    timesteps = retrieve_timesteps(pipe.scheduler, 20)[0]
    if isinstance(timesteps, list):
        timesteps = torch.tensor([int(t) if torch.is_tensor(t) else t for t in timesteps])
    return [int(t.item()) if torch.is_tensor(t) else int(t) for t in timesteps]

def get_timesteps(pipe, timesteps_step_size=5, timesteps_list=None, timesteps_all=False):
    if timesteps_all:
        return get_all_timesteps(pipe)
    
    if timesteps_list:
        return timesteps_list
    
    timesteps = retrieve_timesteps(pipe.scheduler, 20)[0]
    if isinstance(timesteps, list):
        timesteps = torch.tensor([int(t) if torch.is_tensor(t) else t for t in timesteps])

    results = []
    for i, t in enumerate(timesteps):
        if i % timesteps_step_size == 0:
            results.append(int(t.item()) if torch.is_tensor(t) else int(t))
        elif i == len(timesteps) - 1:
            results.append(int(t.item()) if torch.is_tensor(t) else int(t))

    return results

def get_all_layers(pipe):
    """Get all layer indices"""
    transformer_blocks = pipe.transformer_blocks
    return list(range(len(transformer_blocks)))

def get_layers(pipe, layers_step_size=5, layers_list=None, layers_all=False):
    if layers_all:
        return get_all_layers(pipe)
    
    if layers_list:
        return layers_list
    
    transformer_blocks = pipe.transformer_blocks
    num_layers = len(transformer_blocks)
    results = list(range(0, num_layers, layers_step_size))

    if (num_layers - 1) not in results:
        results.append(num_layers - 1)

    return results

def train_probes_online(prompt_dataset, pipe, hooks, num_epochs, batch_size, device):
    """Train probes on the fly by iterating through prompts for multiple epochs.
    
    Uses DataLoader without shuffling. In first epoch, randomly assigns seeds to batches
    and saves them. Subsequent epochs use the saved seeds for reproducibility.
    """
    global training_mode
    
    training_mode = True
    
    # Create DataLoader WITHOUT shuffling
    # Use custom collate_fn to handle string batches
    def collate_fn(batch):
        return batch  # Just return the list of strings as-is
    
    dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    num_batches = len(dataloader)
    
    # Dictionary to store batch_index -> seed mapping
    batch_seeds = {}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch_prompts in enumerate(dataloader):
            # batch_prompts is already a list of strings from our collate_fn
            
            # First epoch: randomly assign seed and save it
            if epoch == 0:
                batch_seed = random.randint(0, 99999)
                batch_seeds[batch_idx] = batch_seed
                print(f"  Batch {batch_idx + 1}/{num_batches} | {len(batch_prompts)} prompts | seed={batch_seed} (assigned)")
            else:
                # Subsequent epochs: use saved seed
                batch_seed = batch_seeds[batch_idx]
                print(f"  Batch {batch_idx + 1}/{num_batches} | {len(batch_prompts)} prompts | seed={batch_seed} (reused)")
            
            generator = torch.Generator(device=device).manual_seed(batch_seed)
            _ = pipe(
                prompt=batch_prompts,
                height=1024,
                width=1024,
                guidance_scale=5.0,
                num_inference_steps=20,
                generator=generator,
            )
        
        print(f"  Epoch {epoch + 1} completed")

def evaluate_probes(prompt_dataset, pipe, hooks, batch_size, device):
    """Evaluate all probes on eval prompts using the same hooks as training.
    
    Metrics are accumulated on-the-fly during forward passes, then aggregated at the end.
    """
    global training_mode, eval_metrics
    
    # Clear previous evaluation metrics
    eval_metrics.clear()
    
    # Set to evaluation mode (hooks will evaluate on-the-fly)
    training_mode = False
    
    # Create DataLoader WITHOUT shuffling (same as training)
    def collate_fn(batch):
        return batch  # Just return the list of strings as-is
    
    dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    num_batches = len(dataloader)
    
    # Dictionary to store batch_index -> seed mapping (same as training)
    batch_seeds = {}
    
    # Run evaluation (hooks will accumulate metrics automatically)
    for batch_idx, batch_prompts in enumerate(dataloader):
        batch_seed = random.randint(0, 99999)
        batch_seeds[batch_idx] = batch_seed
      
        
        print(f"Evaluating batch {batch_idx + 1}/{num_batches} | {len(batch_prompts)} prompts | seed={batch_seed}")
        
        generator = torch.Generator(device=device).manual_seed(batch_seed)
        _ = pipe(
            prompt=batch_prompts,
            height=1024,
            width=1024,
            guidance_scale=5.0,
            num_inference_steps=20,
            generator=generator,
        )
    
    # Aggregate metrics for all probes
    results = []
    for key, probe in probes_dict.items():
        ts, pos, component, grad_type = key
        
        if key not in eval_metrics:
            continue
        
        mae_list = eval_metrics[key]['mae']
        spearman_list = eval_metrics[key]['spearman']
        
        if len(mae_list) == 0:
            continue
        
        # Compute statistics
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        mean_spearman = np.mean(spearman_list) if spearman_list else float('nan')
        var_spearman = np.var(spearman_list) if spearman_list else float('nan')
        
        results.append({
            'timestep': ts,
            'position': pos,
            'component': component,
            'gradient_type': grad_type,
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_spearman': mean_spearman,
            'var_spearman': var_spearman
        })
        
        print(f"  Timestep={ts}, Position={pos}, Component={component}, Gradient={grad_type}: "
              f"MAE={mean_mae:.6f}±{std_mae:.6f}, Spearman={mean_spearman:.6f} (var={var_spearman:.6f})")
    
    return results, batch_seeds

def save_probe(probe, save_dir, ts, pos, component, grad_type, kernel_size):
    """Save a probe model"""
    if pos == "initial":
        folder = os.path.join(save_dir, f"timestep_{ts}", "initial")
    elif pos == "final":
        folder = os.path.join(save_dir, f"timestep_{ts}", "final")
    else:
        folder = os.path.join(save_dir, f"timestep_{ts}", f"layer_{pos:02d}")
        if component != "block_output":
            folder = os.path.join(folder, component)
    
    os.makedirs(folder, exist_ok=True)
    
    filename = f"probe_kernel_{kernel_size}_grad_{grad_type}.pt"
    filepath = os.path.join(folder, filename)
    torch.save(probe.state_dict(), filepath)
    print(f"Saved probe to {filepath}")

def save_eval_image(pred, save_dir, ts, pos, component, grad_type, example_idx, kernel_size):
    """Save evaluation image"""
    if pos == "initial":
        folder = os.path.join(save_dir, f"timestep_{ts}", "initial")
    elif pos == "final":
        folder = os.path.join(save_dir, f"timestep_{ts}", "final")
    else:
        folder = os.path.join(save_dir, f"timestep_{ts}", f"layer_{pos:02d}")
        if component != "block_output":
            folder = os.path.join(folder, component)
    
    os.makedirs(folder, exist_ok=True)
    
    resized_pred = F.interpolate(pred.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False)
    pred_np = resized_pred[0, 0].detach().cpu().numpy()
    
    filename = f"example_{example_idx}_kernel_{kernel_size}_grad_{grad_type}.png"
    filepath = os.path.join(folder, filename)
    plt.imsave(filepath, pred_np, cmap="viridis", format='png')
    print(f"Saved image to {filepath}")

def make_image_saving_hook(position, component_type, timesteps, gradient_types, save_dir, kernel_size, 
                          use_input=False, max_images=5):
    """Unified hook that saves images for evaluation.
    
    Args:
        position: Position identifier (layer index, "initial", or "final")
        component_type: Component type identifier
        timesteps: List of timesteps to save images for
        gradient_types: List of gradient types to save
        save_dir: Directory to save images
        kernel_size: Kernel size for probe
        use_input: If True, use inp[0] instead of out (for proj_out)
        max_images: Maximum number of images to save per gradient type
    """
    example_counter = defaultdict(int)  # Track example index per key
    
    def hook_fn(mod, inp, out):
        global current_timestep, normalize_latents_with_layer_norm, layer_norm
        
        if current_timestep not in timesteps:
            return
        
        # Get the tensor (either from output or input)
        if use_input:
            if not isinstance(inp[0], torch.Tensor):
                return
            tensor = inp[0]
        else:
            if not isinstance(out, torch.Tensor):
                return
            tensor = out
        
        # Reshape tensor to [B, C, H, W]
        if tensor.dim() == 3:
            b, l, d = tensor.shape
            shaped_tensor = tensor.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
        elif tensor.dim() == 4:
            shaped_tensor = tensor
        else:
            return
        
        # Extract guided activations
        mid = shaped_tensor.shape[0] // 2
        guided = shaped_tensor[mid:].detach()
        
        # Normalize if needed
        if normalize_latents_with_layer_norm and layer_norm is not None:
            B, C, H, W = guided.shape
            guided = guided.permute(0, 2, 3, 1)
            with torch.no_grad():
                guided = layer_norm(guided)
            guided = guided.permute(0, 3, 1, 2)
        
        # Save images for each gradient type
        for grad_type in gradient_types:
            key = (current_timestep, position, component_type, grad_type)
            if key not in probes_dict:
                continue
            
            probe = probes_dict[key]
            probe.eval()
            
            with torch.no_grad():
                output = probe(guided)
                # Save only min(max_images, batch_size) images
                num_to_save = min(max_images, output.shape[0])
                for j in range(num_to_save):
                    example_idx = example_counter[key]
                    save_eval_image(output[j:j+1], save_dir, current_timestep, position, 
                                  component_type, grad_type, example_idx, kernel_size)
                    example_counter[key] += 1
    
    return hook_fn

def save_eval_images(prompt_dataset, pipe, hooks, batch_size, device, batch_seeds, 
                    timesteps, positions, components, gradient_types, use_contributions_mode,
                    train_on_block_output, hook_patch_embed, hook_proj_out, save_dir, kernel_size):
    """Save evaluation images by running first batch with image-saving hooks"""
    global training_mode
    
    # Remove all train/eval hooks before registering image-saving hooks
    print("Removing train/eval hooks...")
    for hook in hooks:
        hook.remove()
    
    # Set to evaluation mode
    training_mode = False
    
    # Register image-saving hooks for all relevant components
    print("Registering image-saving hooks...")
    image_hooks = []
    image_hooks += register_time_step_hook(pipe.transformer)
    
    max_images = min(5, batch_size)
    
    if hook_patch_embed:
        image_hooks.append(pipe.transformer.patch_embed.register_forward_hook(
            make_image_saving_hook("initial", "patch_embed", timesteps, gradient_types, 
                                  save_dir, kernel_size, use_input=False, max_images=max_images)))
    
    if hook_proj_out:
        image_hooks.append(pipe.transformer.proj_out.register_forward_hook(
            make_image_saving_hook("final", "proj_out", timesteps, gradient_types, 
                                  save_dir, kernel_size, use_input=True, max_images=max_images)))
    
    if train_on_block_output:
        for i in positions:
            if isinstance(i, int):  # It's a layer index
                image_hooks.append(pipe.transformer.transformer_blocks[i].register_forward_hook(
                    make_image_saving_hook(i, "block_output", timesteps, gradient_types, 
                                          save_dir, kernel_size, use_input=False, max_images=max_images)))
    else:
        for component in components:
            if component in ["patch_embed", "proj_out"]:
                continue
            transformer_blocks = pipe.transformer.transformer_blocks
            for i in positions:
                if isinstance(i, int):  # It's a layer index
                    block = transformer_blocks[i]
                    comp = component
                    if use_contributions_mode:
                        if component == "self_attn":
                            comp = "self_attn_after_gate"
                        elif component == "mix_ffn":
                            comp = "mix_ffn_after_gate"
                    
                    if comp == "self_attn" or comp == "self_attn_after_gate":
                        if use_contributions_mode or comp == "self_attn_after_gate":
                            image_hooks.append(block.identity_after_attn.register_forward_hook(
                                make_image_saving_hook(i, "self_attn_after_gate", timesteps, gradient_types, 
                                                      save_dir, kernel_size, use_input=False, max_images=max_images)))
                        else:
                            image_hooks.append(block.attn1.register_forward_hook(
                                make_image_saving_hook(i, "self_attn", timesteps, gradient_types, 
                                                      save_dir, kernel_size, use_input=False, max_images=max_images)))
                    elif comp == "cross_attn":
                        image_hooks.append(block.attn2.register_forward_hook(
                            make_image_saving_hook(i, "cross_attn", timesteps, gradient_types, 
                                                  save_dir, kernel_size, use_input=False, max_images=max_images)))
                    elif comp == "mix_ffn" or comp == "mix_ffn_after_gate":
                        if use_contributions_mode or comp == "mix_ffn_after_gate":
                            image_hooks.append(block.identity_after_ff.register_forward_hook(
                                make_image_saving_hook(i, "mix_ffn_after_gate", timesteps, gradient_types, 
                                                      save_dir, kernel_size, use_input=False, max_images=max_images)))
                        else:
                            image_hooks.append(block.ff.register_forward_hook(
                                make_image_saving_hook(i, "mix_ffn", timesteps, gradient_types, 
                                                      save_dir, kernel_size, use_input=False, max_images=max_images)))
    
    # Run only first batch with its original seed
    def collate_fn(batch):
        return batch
    
    dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    first_batch = next(iter(dataloader))
    first_batch_seed = batch_seeds[0]
    
    print(f"Saving images for first batch (seed={first_batch_seed})...")
    
    generator = torch.Generator(device=device).manual_seed(first_batch_seed)
    _ = pipe(
        prompt=first_batch,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_inference_steps=20,
        generator=generator,
    )
    
    # Remove image-saving hooks
    for hook in image_hooks:
        hook.remove()
    
    print("Image saving completed.")

def save_results_to_csv(results, output_path, kernel_size, unique_id):
    """Save evaluation results to CSV with mean, std MAE and mean, var Spearman"""
    # Add unique identifier to filename
    base_name = os.path.splitext(output_path)[0]
    ext = os.path.splitext(output_path)[1] or '.csv'
    output_path = f"{base_name}_{unique_id}{ext}"
    
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestep', 'position', 'component', 'gradient_type', 'kernel_size', 
                         'mean_mae', 'std_mae', 'mean_spearman', 'var_spearman'])
        for r in results:
            writer.writerow([
                r['timestep'],
                r['position'],
                r['component'],
                r['gradient_type'],
                kernel_size,
                r['mean_mae'],
                r['std_mae'],
                r['mean_spearman'],
                r['var_spearman']
            ])
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train probes online during forward pass")
    parser.add_argument("--n_train", type=int, required=True, help="Number of training prompts")
    parser.add_argument("--n_eval", type=int, required=True, help="Number of evaluation prompts")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for probe")
    parser.add_argument("--gradient_types", type=str, nargs="+", default=["Vertical"], 
                        choices=["Vertical", "Horizontal", "Gaussian"], help="Gradient types to train probes for")
    parser.add_argument("--layers_step", type=int, default=-1, help="Layer step size")
    parser.add_argument("--timesteps_step", type=int, default=-1, help="Timestep step size")
    parser.add_argument("--component_types", type=str, nargs="+", 
                        choices=["self_attn", "cross_attn", "mix_ffn"], 
                        default=["mix_ffn"], 
                        help="Component types to train probes for (can specify multiple)")
    parser.add_argument("--timesteps", type=str, nargs="+", default=[], help="Specific timesteps or 'all'")
    parser.add_argument("--layers", type=str, nargs="+", default=[], help="Specific layers or 'all'")
    parser.add_argument("--use_contributions_mode", action="store_true", help="Use contributions mode")
    parser.add_argument("--normalize_latents_with_layer_norm", action="store_true", help="Normalize latents with layer norm")
    parser.add_argument("--train_on_block_output", action="store_true", help="Train on entire block output instead of individual components")
    parser.add_argument("--hook_patch_embed", action="store_true", help="Hook to patch_embed output (initial representation)")
    parser.add_argument("--hook_proj_out", action="store_true", help="Hook to proj_out input (final representation)")
    parser.add_argument("--save_probes", action="store_true", help="Save probe models as .pt files")
    parser.add_argument("--save_images", action="store_true", help="Save evaluation images")
    parser.add_argument("--probes_output_dir", type=str, default="probes", help="Directory to save probes")
    parser.add_argument("--images_output_dir", type=str, default="eval_images", help="Directory to save evaluation images")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    # Validate arguments
    has_timesteps = len(args.timesteps) > 0
    has_timesteps_step = args.timesteps_step > 0
    if not has_timesteps and not has_timesteps_step:
        raise ValueError("Must specify either --timesteps (with values or 'all') or --timesteps_step")
    if has_timesteps and has_timesteps_step:
        raise ValueError("Cannot use both --timesteps and --timesteps_step")
    
    has_layers = len(args.layers) > 0
    has_layers_step = args.layers_step > 0
    if not has_layers and not has_layers_step and not args.train_on_block_output and not args.hook_patch_embed and not args.hook_proj_out:
        raise ValueError("Must specify either --layers (with values or 'all') or --layers_step, or use --train_on_block_output/--hook_patch_embed/--hook_proj_out")
    if has_layers and has_layers_step:
        raise ValueError("Cannot use both --layers and --layers_step")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate unique identifier
    unique_id = str(uuid.uuid4())[:8]
    
    # Load model
    print("Loading model...")
    pipe = load_model()
    
    # Setup layer norm if needed
    global layer_norm, normalize_latents_with_layer_norm
    normalize_latents_with_layer_norm = args.normalize_latents_with_layer_norm
    if normalize_latents_with_layer_norm:
        layer_norm = pipe.transformer.norm_out.norm.to(device)
    
    # Parse timesteps
    timesteps_all = "all" in args.timesteps if args.timesteps else False
    timesteps_list = None
    if args.timesteps and not timesteps_all:
        timesteps_list = [int(t) for t in args.timesteps if t != "all"]
    
    # Parse layers
    layers_all = "all" in args.layers if args.layers else False
    layers_list = None
    if args.layers and not layers_all:
        layers_list = [int(l) for l in args.layers if l != "all"]
    
    # Get timesteps and layers
    timesteps = get_timesteps(pipe, args.timesteps_step, timesteps_list, timesteps_all)
    layers = get_layers(pipe.transformer, args.layers_step, layers_list, layers_all)
    
    print(f"Timesteps: {timesteps}")
    print(f"Layers: {layers}")
    print(f"Component types: {args.component_types}")
    print(f"Gradient types: {args.gradient_types}")
    
    # Determine components and positions configuration
    components_config = {}
    
    # Add layer positions
    if args.train_on_block_output:
        components_config["block_output"] = layers
    else:
        # Handle multiple component types
        for comp_type in args.component_types:
            comp = comp_type
            if args.use_contributions_mode:
                if comp_type == "self_attn":
                    comp = "self_attn_after_gate"
                elif comp_type == "mix_ffn":
                    comp = "mix_ffn_after_gate"
            # If component already exists, extend layers (shouldn't happen, but safe)
            if comp in components_config:
                components_config[comp].extend(layers)
                components_config[comp] = list(set(components_config[comp]))  # Remove duplicates
            else:
                components_config[comp] = layers
    
    # Add initial/final positions if requested
    if args.hook_patch_embed:
        components_config["patch_embed"] = ["initial"]
    
    if args.hook_proj_out:
        components_config["proj_out"] = ["final"]
    
    # Create flat lists for evaluation function
    components = list(components_config.keys())
    positions = []
    for comp, comp_positions in components_config.items():
        positions.extend(comp_positions)
    positions = list(set(positions))  # Remove duplicates
    
    print(f"Components configuration: {components_config}")
    print(f"Total probes to initialize: {sum(len(comp_positions) * len(args.gradient_types) * len(timesteps) for comp_positions in components_config.values())}")
    
    # Install contributions mode if needed
    if args.use_contributions_mode:
        for i, block in enumerate(pipe.transformer.transformer_blocks):
            install_forward_with_identities(block)
            print(f"Patched transformer block {i}")
    
    # Set global random seed for prompt sampling
    random.seed(args.seed)
    
    # Sample prompts (now that seed is set)
    print("Sampling prompts...")
    train_prompts, eval_prompts = sample_prompts(args.n_train, args.n_eval)
    print(f"Training prompts: {len(train_prompts)}")
    print(f"Evaluation prompts: {len(eval_prompts)}")
    
    # Create datasets
    train_dataset = PromptDataset(train_prompts)
    eval_dataset = PromptDataset(eval_prompts)
    
    # Get sample activation shape
    print("Getting sample activation shape...")
    sample_shape = None
    def get_shape_hook(mod, inp, out):
        nonlocal sample_shape
        if out.dim() == 3:
            b, l, d = out.shape
            shaped_out = out.unflatten(1, (int(l**0.5), int(l**0.5))).permute(0, 3, 1, 2)
        elif out.dim() == 4:
            shaped_out = out
        else:
            return
        mid = shaped_out.shape[0] // 2
        guided = shaped_out[mid:]
        if sample_shape is None:
            sample_shape = guided.shape[1:]  # (C, H, W)
    
    # Try to get shape from first available component
    temp_hook = None
    if args.hook_patch_embed:
        temp_hook = pipe.transformer.patch_embed.register_forward_hook(get_shape_hook)
    elif layers:
        block = pipe.transformer.transformer_blocks[layers[0]]
        if args.train_on_block_output:
            temp_hook = block.register_forward_hook(get_shape_hook)
        else:
            # Use first component type to get shape (all should have same shape)
            first_comp_type = args.component_types[0]
            if first_comp_type == "mix_ffn":
                if args.use_contributions_mode:
                    temp_hook = block.identity_after_ff.register_forward_hook(get_shape_hook)
                else:
                    temp_hook = block.ff.register_forward_hook(get_shape_hook)
            elif first_comp_type == "self_attn":
                if args.use_contributions_mode:
                    temp_hook = block.identity_after_attn.register_forward_hook(get_shape_hook)
                else:
                    temp_hook = block.attn1.register_forward_hook(get_shape_hook)
            elif first_comp_type == "cross_attn":
                temp_hook = block.attn2.register_forward_hook(get_shape_hook)
    
    sample_seed = random.randint(0, 99999)
    _ = pipe(
        prompt=[train_prompts[0]],
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_inference_steps=20,
        generator=torch.Generator(device=device).manual_seed(sample_seed),
    )
    
    if temp_hook:
        temp_hook.remove()
    
    if sample_shape is None:
        raise RuntimeError("Could not determine activation shape")
    
    print(f"Sample activation shape: {sample_shape}")
    
    # Initialize all probes
    print("Initializing probes...")
    initialize_all_probes(timesteps, components_config, args.kernel_size, args.gradient_types, device, (1,) + sample_shape)
    
    # Register hooks
    print("Registering hooks...")
    hooks = []
    hooks += register_time_step_hook(pipe.transformer)
    
    if args.hook_patch_embed:
        hooks += register_patch_embed_hook(pipe.transformer, timesteps, args.gradient_types)
    
    if args.hook_proj_out:
        hooks += register_proj_out_hook(pipe.transformer, timesteps, args.gradient_types)
    
    if not args.train_on_block_output and layers:
        for component in [c for c in components if c not in ["patch_embed", "proj_out"]]:
            hooks += register_component_hooks(pipe.transformer, layers, timesteps, component, 
                                            args.use_contributions_mode, args.gradient_types, False)
    elif args.train_on_block_output and layers:
        hooks += register_component_hooks(pipe.transformer, layers, timesteps, None, 
                                        args.use_contributions_mode, args.gradient_types, True)
    
    # Training
    print("Starting training...")
    train_probes_online(train_dataset, pipe, hooks, args.num_epochs, args.batch_size, device)
    
    # Evaluation (uses same hooks as training, just set training_mode=False)
    print("Starting evaluation...")
    results, batch_seeds = evaluate_probes(eval_dataset, pipe, hooks, args.batch_size, device)
    
    # Save probes
    if args.save_probes:
        print("Saving probes...")
        for key, probe in probes_dict.items():
            ts, pos, component, grad_type = key
            save_probe(probe, args.probes_output_dir, ts, pos, component, grad_type, args.kernel_size)
    
    # Save evaluation images
    if args.save_images:
        print("Saving evaluation images...")
        save_eval_images(eval_dataset, pipe, hooks, args.batch_size, device, batch_seeds,
                        timesteps, positions, components, args.gradient_types, args.use_contributions_mode,
                        args.train_on_block_output, args.hook_patch_embed, args.hook_proj_out,
                        args.images_output_dir, args.kernel_size)
    
    # Save results
    print("Saving results...")
    save_results_to_csv(results, args.output_csv, args.kernel_size, unique_id)
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    print("Done!")

if __name__ == "__main__":
    main()

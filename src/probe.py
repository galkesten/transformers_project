import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import re
import os
import argparse
import csv
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import torch.nn.functional as F

# === Fixed Training Hyperparameters ===
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_EPOCHS = 10
DEFAULT_STEP_SIZE = 5
DEFAULT_GAMMA = 0.5
DEFAULT_GRADIENT_TYPE = "Vertical"

# === Dataset ===
class LatentTimestepDataset(Dataset):
    def __init__(self, data, timestep):
        self.latents = [entry['latents'] for entry in data]
        self.timestep = timestep

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]

def extract_timestep_from_filename(file_path):
    match = re.search(r'timestep_(\d+)', os.path.basename(file_path))
    if not match:
        raise ValueError(f"Timestep not found in filename: {file_path}")
    return int(match.group(1))

def load_dataset(file_path):
    data = torch.load(file_path)
    timestep = extract_timestep_from_filename(file_path)
    return LatentTimestepDataset(data, timestep)

# === Gradient Map Generator ===
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

# === Probe + Target Creation ===
def create_probe_and_target(train_loader, kernel_size, gradient_type, device):
    sample = next(iter(train_loader)).to(device)
    B, C, H, W = sample.shape
    probe = nn.Conv2d(C, 1, kernel_size=(kernel_size, kernel_size), stride=1, padding=0).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    with torch.no_grad():
        output = probe(sample)
    _, _, out_H, out_W = output.shape
    target = generate_gradient_map(out_H, out_W, gradient_type).unsqueeze(0).unsqueeze(0).to(device)

    return probe, target

# === Training Loop ===
def train_probe(probe, train_loader, target, device):
    optimizer = torch.optim.Adam(probe.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DEFAULT_STEP_SIZE, gamma=DEFAULT_GAMMA)
    loss_fn = nn.MSELoss()

    for epoch in range(DEFAULT_NUM_EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B = batch.size(0)
            optimizer.zero_grad()
            output = probe(batch)
            loss = loss_fn(output, target.expand(B, -1, -1, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{DEFAULT_NUM_EPOCHS} | Loss: {total_loss:.4f}")

# === Evaluation ===
def evaluate_probe(probe, test_loader, target, save_dir, mae_out_path, spearman_out_path, gradient_type, device, num_examples_to_save=5):
    os.makedirs(save_dir, exist_ok=True)
    probe.eval()
    probe.to(device)
    total_mae = 0.0
    spearman_scores = []

    timestep = getattr(test_loader.dataset, "timestep", "unknown")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            B = batch.size(0)
            preds = probe(batch)
            truth = target.expand(B, -1, -1, -1)

            for j in range(B):
                pred_flat = preds[j].flatten().cpu().numpy()
                target_flat = truth[j].flatten().cpu().numpy()

                mae = np.abs(pred_flat - target_flat).mean()
                total_mae += mae

                corr, _ = spearmanr(pred_flat, target_flat)
                if not np.isnan(corr):
                    spearman_scores.append(corr)

                if i * B + j < num_examples_to_save:
                    resized_pred = F.interpolate(preds[j:j+1], size=(512, 512), mode="bilinear", align_corners=False)
                    pred_np = resized_pred[0, 0].detach().cpu().numpy()
                    save_path = os.path.join(save_dir, f"timestep_{timestep}_kernel_{probe.kernel_size[0]}_example_{i * B + j}_{gradient_type}.png")
                    plt.imsave(save_path, pred_np, cmap="viridis", format='png')

    avg_mae = total_mae / len(test_loader.dataset)
    avg_spearman = np.mean(spearman_scores) if spearman_scores else float('nan')

    write_csv_line(mae_out_path, ["timestep", "kernel", "gradient_type", "mae"],
               [timestep, probe.kernel_size[0], gradient_type, avg_mae])
    write_csv_line(spearman_out_path, ["timestep", "kernel", "gradient_type", "spearman"],
                [timestep, probe.kernel_size[0], gradient_type, avg_spearman])

    print(f"[EVAL] MAE: {avg_mae:.6f} | Spearman: {avg_spearman:.6f}")

# === CSV Logger ===
def write_csv_line(path, header, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Train linear probe on latent features")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--models_output_folder", type=str, required=True)
    parser.add_argument("--test_output_folder", type=str, required=True)
    parser.add_argument("--test_results_file_path_mae", type=str, required=True)
    parser.add_argument("--test_results_file_path_spearman", type=str, required=True)
    parser.add_argument("--gradient_type", type=str, default="Vertical", choices=["Vertical", "Horizontal", "Gaussian"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(args.train_path)
    test_dataset = load_dataset(args.test_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} samples")
    first_batch = next(iter(train_loader))
    print(f"Train batch shape: {first_batch.shape}")

    probe, target = create_probe_and_target(train_loader, args.kernel_size, args.gradient_type, device)
    train_probe(probe, train_loader, target, device)

    os.makedirs(args.models_output_folder, exist_ok=True)
    timestep = getattr(train_dataset, "timestep", "unknown")
    model_path = os.path.join(args.models_output_folder, f"probe_timestep_{timestep}_kernel_{args.kernel_size}_grad_{args.gradient_type}.pt")
    torch.save(probe.state_dict(), model_path)
    print(f"[SAVE] Probe model saved to {model_path}")

    evaluate_probe(probe=probe, test_loader=test_loader, target=target, save_dir=args.test_output_folder,
                   mae_out_path=args.test_results_file_path_mae, spearman_out_path=args.test_results_file_path_spearman,
                   gradient_type=args.gradient_type, device=device)

if __name__ == "__main__":
    main()

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
from filelock import FileLock

# === Fixed Training Hyperparameters ===
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_EPOCHS = 20
DEFAULT_STEP_SIZE = 30
DEFAULT_GAMMA = 0.1
DEFAULT_GRADIENT_TYPE = "Vertical"


def extract_accumulate_number(filename):
    # This assumes your pattern is timestep_XXX_accumulate_N.pt
    m = re.search(r'_accumulate_(\d+)\.pt$', filename)
    return int(m.group(1)) if m else -1


# === Dataset ===
class LatentTimestepDataset(Dataset):
    def __init__(self, data, timestep):
        self.latents = [entry['latents'] for entry in data]
        self.timestep = timestep

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]

class LatentFlexibleDataset(Dataset):
    def __init__(
        self, 
        folder, 
        timestep, 
        accumulate_mode=False, 
        latent_type='latents', 
        accumulate_size=500
    ):
        """
        folder: directory containing .pt files
        timestep: which timestep to load (int)
        accumulate_mode: True if using accumulate files, else single file per timestep
        latent_type: which latent to return ('latents', 'guided', 'unguided', or 'both')
        accumulate_size: number of examples per accumulate file (except last)
        """
        self.folder = os.path.expanduser(folder)
        self.timestep = timestep
        self.accumulate_mode = accumulate_mode
        self.latent_type = latent_type

        if not accumulate_mode:
            file = os.path.join(self.folder, f'timestep_{timestep}.pt')
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found.")
            self.data = torch.load(file, map_location='cpu')
            self.length = len(self.data)
        else:
            self.file_list = sorted(
                [os.path.join(self.folder, f)
                for f in os.listdir(self.folder)
                if f.startswith(f'timestep_{timestep}_accumulate_') and f.endswith('.pt')],
                key=extract_accumulate_number
            )
            N = len(self.file_list)
            if N == 0:
                raise FileNotFoundError(f"No accumulate files found for timestep {timestep} in {self.folder}")

            if N == 1:
                first_len = last_len = len(torch.load(self.file_list[0], map_location='cpu'))
            else:
                first_len = accumulate_size  # Assume
                last_len = accumulate_size   # Assume
                try:
                    first_len = len(torch.load(self.file_list[0], map_location='cpu'))
                    last_len = len(torch.load(self.file_list[-1], map_location='cpu'))
                except Exception:
                    pass  # fallback to provided accumulate_size

            self.lengths = [first_len] * (N - 1) + [last_len]
            self.offsets = [0]
            for l in self.lengths:
                self.offsets.append(self.offsets[-1] + l)
            self.total = self.offsets[-1]
            self._cached_file_idx = None
            self._cached_data = None

    def __len__(self):
        if not self.accumulate_mode:
            return self.length
        return self.total

    def __getitem__(self, idx):
        if not self.accumulate_mode:
            #print(f"[DEBUG] Single-file mode: Loading index {idx} from self.data")
            entry = self.data[idx]
        else:
            file_idx = None
            for i in range(len(self.offsets) - 1):
                if self.offsets[i] <= idx < self.offsets[i + 1]:
                    file_idx = i
                    break
            if file_idx is None:
                raise IndexError(f"Index {idx} out of range! Offsets: {self.offsets}")

            local_idx = idx - self.offsets[file_idx]
            #print(f"[DEBUG] Accumulate mode: idx {idx} is in file {file_idx} ({self.file_list[file_idx]}), local_idx {local_idx}")
            if self._cached_file_idx != file_idx:
                print(f"[DEBUG] Loading new file into cache: {self.file_list[file_idx]}", flush=True)
                self._cached_data = torch.load(self.file_list[file_idx], map_location='cpu')
                self._cached_file_idx = file_idx
            entry = self._cached_data[local_idx]

        # Add a print for what kind of tensor is being returned
        #print(f"[DEBUG] Returning latent_type: {self.latent_type}")
        if self.latent_type == 'latents':
            return entry['latents']
        elif self.latent_type == 'guided':
            return entry['guided']
        elif self.latent_type == 'unguided':
            return entry['unguided']
        elif self.latent_type == 'both':
            return torch.cat([entry['guided'], entry['unguided']], dim=0)
        else:
            raise ValueError(f"Unknown latent_type: {self.latent_type}")


        

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
    sample = next(iter(train_loader)).to(device).float()
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
        i = 0
        for batch in train_loader:
            i = i+1 
            batch = batch.to(device).float()
            #print("Latents: min", batch.min().item(), "max",batch.max().item(), "mean", batch.mean().item(), "std", batch.std().item())
            #print("Target: min", target.min().item(), "max", target.max().item(), "mean", target.mean().item(), "std", target.std().item())
            B = batch.size(0)
            optimizer.zero_grad()
            output = probe(batch)
            loss = loss_fn(output, target.expand(B, -1, -1, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / i
        print(f"Epoch {epoch+1}/{DEFAULT_NUM_EPOCHS} | Avg Loss per batch: {avg_loss:.4f}", flush=True)

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
            batch = batch.to(device).float()
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

    print(f"[EVAL] MAE: {avg_mae:.6f} | Spearman: {avg_spearman:.6f}", flush=True)

def write_csv_line(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Train linear probe on latent features")
    parser.add_argument("--latents_folder_train", type=str, required=True)
    parser.add_argument("--latents_folder_test", type=str, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--accumulate_mode", action="store_true", help="Use accumulate files")
    parser.add_argument("--latent_type", type=str, default="latents", choices=["latents", "guided", "unguided", "both"])
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--models_output_folder", type=str, required=True)
    parser.add_argument("--test_output_folder", type=str, required=True)
    parser.add_argument("--test_results_file_path_mae", type=str, required=True)
    parser.add_argument("--test_results_file_path_spearman", type=str, required=True)
    parser.add_argument("--gradient_type", type=str, default="Vertical", choices=["Vertical", "Horizontal", "Gaussian"])
    parser.add_argument("--accumulate_size", type=int, default=500, help="Number of samples per accumulate file (default 500)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LatentFlexibleDataset(args.latents_folder_train, args.timestep, args.accumulate_mode, args.latent_type)
    test_dataset = LatentFlexibleDataset(args.latents_folder_test, args.timestep, args.accumulate_mode, args.latent_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} samples", flush=True)
    first_batch = next(iter(train_loader))
    print(f"Train batch shape: {first_batch.shape}", flush=True)

    print(f"Test dataset: {len(test_dataset)} samples", flush=True)
    first_batch = next(iter(test_loader))
    print(f"Test batch shape: {first_batch.shape}", flush=True)

    probe, target = create_probe_and_target(train_loader, args.kernel_size, args.gradient_type, device)
    train_probe(probe, train_loader, target, device)

    os.makedirs(args.models_output_folder, exist_ok=True)
    timestep = getattr(train_dataset, "timestep", "unknown")
    model_path = os.path.join(args.models_output_folder, f"probe_timestep_{timestep}_kernel_{args.kernel_size}_grad_{args.gradient_type}.pt")
    torch.save(probe.state_dict(), model_path)
    print(f"[SAVE] Probe model saved to {model_path}", flush=True)

    evaluate_probe(probe=probe, test_loader=test_loader, target=target, save_dir=args.test_output_folder,
                   mae_out_path=args.test_results_file_path_mae, spearman_out_path=args.test_results_file_path_spearman,
                   gradient_type=args.gradient_type, device=device)

if __name__ == "__main__":
    main()

import os

folder = os.path.expanduser('~/sana_outputs/train/post_layer_norm_latents')
for fname in os.listdir(folder):
    if fname.endswith('.pt'):
        full = os.path.join(folder, fname)
        sz = os.path.getsize(full)
        print(f"{fname:30} {sz/1024/1024:.2f} MB")

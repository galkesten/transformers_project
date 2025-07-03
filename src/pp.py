import os

folder = os.path.expanduser('sana_outputs/test/post_layer_norm_latents')
total_bytes = 0

for fname in os.listdir(folder):
    full = os.path.join(folder, fname)
    if os.path.isfile(full):
        total_bytes += os.path.getsize(full)

print(f"Total: {total_bytes/1024/1024/1024:.2f} GB ({total_bytes/1024/1024:.2f} MB)")

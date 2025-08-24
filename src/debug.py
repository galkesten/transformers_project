
import torch, diffusers, transformers, xformers
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("xformers:", xformers.__version__)
try:
    import triton; print("triton:", triton.__version__)
except Exception as e:
    print("triton: NOT AVAILABLE ->", e)


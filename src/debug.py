import torch

load = torch.load("mean_ablations/mean_activations_mix_ffn.pt")
print(load)
are_there_nans = torch.isnan(load).any()
print(are_there_nans)
print(load.shape)
#print if there are infs
print(torch.isinf(load).any())
#print if there are nans
print(torch.isnan(load).any())
#print if there are infs or nans
print(torch.isinf(load).any() or torch.isnan(load).any())
#print if there are infs or nans
print(load.to(torch.float16).isnan().any())
print(load.to(torch.float16).isinf().any())
print(load.to(torch.float16).isnan().any() or load.to(torch.float16).isinf().any())







import torch
def rmsnorm_reference(x,weight,eps=1e-6):
    normalized =  x * torch.rsqrt((x*x).mean(dim=-1,keepdim=True) + eps)
    return normalized * weight

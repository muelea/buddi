import torch 

def dict_to_device(batch, device='cuda'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = dict_to_device(v)
        else:
            batch[k] = v
    return batch
import torch


def safe_concat(buffer: list[torch.Tensor], tensors: tuple[torch.Tensor]) -> None:
    for i, tensor in enumerate(tensors):
        while tensor.dim() < 2:
            tensor = tensor.unsqueeze(0)
        buffer[i] = torch.cat((buffer[i], tensor), dim=0)

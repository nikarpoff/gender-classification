import torch


def define_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

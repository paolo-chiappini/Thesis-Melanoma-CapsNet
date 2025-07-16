import torch
import sys


def get_device(cpu_override=False):
    multi_gpu = False

    if cpu_override:
        return torch.device("cpu"), False

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        multi_gpu = torch.cuda.device_count() > 1
        torch.cuda.empty_cache()
        return device, multi_gpu

    elif sys.platform.startswith("win"):
        try:
            import torch_directml
        except ImportError:
            raise ImportError(
                "DirectML is not installed. Please install it to use this mode."
            )
        if torch_directml.is_available() and torch_directml.device_count() > 0:
            device = torch_directml.device()
            multi_gpu = torch_directml.device_count() > 1
            return device, multi_gpu
        else:
            raise RuntimeError("No compatible GPU found (CUDA or DirectML).")

    raise RuntimeError("No compatible GPU found (CUDA or DirectML).")

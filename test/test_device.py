import torch
import torch_directml

multi_gpu = False
# Try CUDA
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    RUN_MODE = 'cuda'
    device = torch.device('cuda')
    multi_gpu = (torch.cuda.device_count() > 1)
# Fallback to directml (for AMD GPU)
elif torch_directml.device_count() > 0:
    RUN_MODE = 'directml'
    device = torch_directml.device()
# No supported device found (prevent running on CPU)
else:
    raise RuntimeError("No compatible GPU found (CUDA or DirectML).")

print(f'''
============================= CONFIG =============================
- Running mode \t\t: ({RUN_MODE})
- Running on device \t: {device}
- Multi GPU \t\t: {multi_gpu}
==================================================================
''')
import os
import torch
import torch_directml
from torchvision import datasets, transforms
from trainer import CapsNetTrainer

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

exit()

DATA_PATH = os.path.join('baseline_capsnet/')


epochs = 50
batch_size = 128
learning_rate = 1e-3
routing_steps = 3
lr_decay = 0.96
classes = range(2) # Benign 0, Malign 1

if (torch.cuda.is_available() and torch.cuda.device_count() > 1): 
    batch_size *= torch.cuda.device_count()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

loaders = {}
trainset = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform)
loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform)
loaders['test'] = torch.utils.data.DataLoader(testset, batch_size, shuffle=True, num_workers=2)

caps_net = CapsNetTrainer(loaders, batch_size, learning_rate, routing_steps, lr_decay, device, multi_gpu)
caps_net.run(epochs, classes)
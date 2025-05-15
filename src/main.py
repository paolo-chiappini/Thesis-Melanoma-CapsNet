import os
import torch
import torch_directml
from torchvision import transforms
from trainers import trainer_simple
import argparse
from utils.loaders import get_dataset

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

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data')
parser.add_argument('--dataset', default='PH2', choices=['PH2', 'ISIC'], help='Dataset to use: PH2 or ISIC')

args = parser.parse_args()
if args.data_root:
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root path does not exist: {args.data_root}")
    else:
        print(f"Using data root path: {args.data_root}")
DATA_PATH = args.data_root

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# switch between datasets
dataset = get_dataset(args.dataset, DATA_PATH, transform=transform)
if dataset is None:
    print(f"Dataset not found: {args.dataset}")
    exit()

epochs = 50
batch_size = 128
learning_rate = 1e-3
routing_steps = 3
lr_decay = 0.96
classes = range(2) # Benign 0, Malignant 1

if (torch.cuda.is_available() and torch.cuda.device_count() > 1): 
    batch_size *= torch.cuda.device_count()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])


def main():
    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=(0 if not multi_gpu else 2), pin_memory=True)

    caps_net = trainer_simple.CapsNetTrainer(loaders, batch_size, learning_rate, routing_steps, lr_decay, device=device, multi_gpu=multi_gpu)
    caps_net.run(epochs, classes)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
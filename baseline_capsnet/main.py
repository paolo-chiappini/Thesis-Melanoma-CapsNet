import os
import torch
from torchvision import datasets, transforms
from trainer import CapsNetTrainer

DATA_PATH = os.path.join('baseline_capsnet/temp')

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

caps_net = CapsNetTrainer(loaders, batch_size, learning_rate, routing_steps, lr_decay)
caps_net.run(epochs, classes)
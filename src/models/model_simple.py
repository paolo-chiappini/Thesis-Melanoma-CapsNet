import torch
import torch.nn as nn
from . import capsules as caps
from numpy import prod

class CapsuleNetwork(nn.Module):
    '''
    Original network architecture from Hinton et al. 
    '''
    def __init__(self, img_shape, channels, primary_dim, num_classes, output_dim, routing_steps, device: torch.device, kernel_size=3):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device
        
        self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size=kernel_size, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Capsules
        self.primary = caps.PrimaryCapsules(channels, channels, primary_dim, kernel_size)
        
        primary_caps = int(channels / primary_dim * (img_shape[1] - 2 * (kernel_size - 1)) * (img_shape[2] - 2 * (kernel_size - 1)) / 4)
        self.digits = caps.RoutingCapsules(primary_dim, primary_caps, num_classes, output_dim, routing_steps, device=self.device)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True), 
            nn.Linear(1024, int(prod(img_shape))),
            nn.Sigmoid()
        )                 
        
    def forward(self, x): 
        out = self.conv1(x)
        out = self.relu(out)
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)
        
        # Reconstruct image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        
        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)
        
        return preds, reconstructions
        
        
               
        
        
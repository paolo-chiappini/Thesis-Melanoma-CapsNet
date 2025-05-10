# credits: https://github.com/danielhavir/capsule-network

import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(s, dim=-1): 
    '''
    Non-linear squashing of input vectors s_j.  
    v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||   (1)
    
    Params: 
    - s:   vector to squash (before activation)
    - dim: dimension over which to take the norm
    
    Returns: 
    - The squashed input vector  
    '''
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

class PrimaryCapsules(nn.Module): 
    def __init__(self, input_channels, output_channels, caps_dim, kernel_size=9, stride=2, padding=0):
        '''
        Params: 
        - input_channels:   number of channels in input.
        - output_channles:  number of channels in output. 
        - caps_dim:         dimension (length) of output vector (number of capsules). 
        ''' 
        super(PrimaryCapsules, self).__init__()
        self.caps_dim = caps_dim
        self._caps_channels = int(output_channels / caps_dim) # number of channels for each capsule
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.conv(x)
        # Reshape output (batch_size, C_caps, H_caps, W_caps, caps_dim)
        x = x.view(x.size(0), self._caps_channels, x.size(2), x.size(3), self.caps_dim)
        # Flatten outpus into (batch_size, C_caps * H_caps * W_caps, caps_dim)
        x = x.view(x.size(0), -1, self.caps_dim)
        return squash(x)
    
    
class RoutingCapsules(nn.Module):
    def __init__(self, input_dim, input_caps, num_caps, caps_dim, routing_steps, device: torch.device):
        '''
        Params: 
        - input_dim:        dimension (length) of capsule vectors.
        - input_caps:       number of capsules in input to the layer. 
        - num_caps:         number of capsules in the layer.
        - caps_dim:         dimension (length) of output vector (number of capsules). 
        - routing_steps:    number of iterations of the routing algorithm.
        '''
        super(RoutingCapsules, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.routing_steps = routing_steps
        self.device = device 
        # Random initialization of the W matrix
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, input_caps, caps_dim, input_dim))
        
    def __repr__(self):
        tab = '\t'
        line = '\n'
        res = f"{self.__class__.__name__}({line}"
        res += f"{tab}(0): CapsuleLinear({self.input_dim}, {self.caps_dim}){line}"
        res += f"{tab}(1): Routing(routing_steps={self.routing_steps}){line})"
        return res

    
    def forward(self, x): 
        batch_size = x.size(0)
        # Unsqueeze to (batch_size, 1, input_caps, input_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        
        # W @ x = (1, num_caps, input_caps, caps_dim, input_dim) @ (batch_size, 1, input_caps, input_dim, 1) = (batch_size, num_caps, input_caps, caps_dim, 1)
        u_hat = torch.matmul(self.W, x)
        # Remove last dimension (batch_size, num_caps, input_caps, caps_dim)
        u_hat = u_hat.squeeze(-1)
        # Prevent gradient from flowing during routing iterations
        detached_u_hat = u_hat.detach()
        
        '''
        Procedure 1 Routing algorithm.  
        1: procedure ROUTING(u^j|i, r, l)
        2:      for all capsule i in layer l and capsule j in layer (l + 1): bij ← 0. 
        3:      for r iterations do 
        4:          for all capsule i in layer l: ci ← softmax(bi) . softmax computes Eq. 3 
        5:          for all capsule j in layer (l + 1): sj ← ∑  i cij u^j|i  
        6:          for all capsule j in layer (l + 1): vj ← squash(sj) . squash computes Eq. 1 
        7:          for all capsule i in layer l and capsule j in layer (l + 1): bij ← bij + u^j|i.vj  
                return vj
        '''
        b = torch.zeros(batch_size, self.num_caps, self.input_caps, 1).to(self.device)
        for r in range(self.routing_steps - 1):
            c = F.softmax(b, dim=1)
            s = (c * detached_u_hat).sum(dim=2)
            v = squash(s)
            b += torch.matmul(detached_u_hat, v.unsqueeze(-1))
            
        # Perform last iteration on attached u_hat
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)
        
        return v
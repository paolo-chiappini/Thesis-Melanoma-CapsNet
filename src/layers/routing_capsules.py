import torch
import torch.nn as nn
from sparsemax import Sparsemax

from utils.functional import squash


class RoutingCapsules(nn.Module):
    def __init__(
        self,
        input_dim,
        num_input_caps,
        num_class_caps,
        capsule_dimension,
        routing_steps,
        device: torch.device,
        routing_algorithm: str = "softmax",
    ):
        """
        Params:
        - input_dim:             dimension (length) of capsule vectors.
        - num_input_caps:        number of capsules in input to the layer.
        - num_class_caps:        number of capsules in the layer (one per class).
        - capsule_dimension:     dimension (length) of output vector.
        - routing_steps:         number of iterations of the routing algorithm.
        - routing_algorithm:     routing algorithm to use. Either 'sigmoid' or 'softmax'.
        """
        super(RoutingCapsules, self).__init__()
        self.input_dim = input_dim
        self.num_input_caps = num_input_caps
        self.num_class_caps = num_class_caps
        self.capsule_dimension = capsule_dimension
        self.routing_steps = routing_steps
        self.device = device
        self.routing_algorithm = routing_algorithm

        # Random initialization of the W matrix
        self.W = nn.Parameter(
            0.01
            * torch.randn(
                1, num_class_caps, num_input_caps, capsule_dimension, input_dim
            )
        )

        self.routing_activation = None
        if self.routing_algorithm == "sigmoid":
            self.routing_activation = (
                nn.Sigmoid()
            )  # Sigmoid routing like in LaLonde et al.
        elif self.routing_algorithm == "softmax":
            self.routing_activation = nn.Softmax(dim=1)
        elif self.routing_algorithm == "sparsemax":
            self.routing_activation = Sparsemax(dim=1)
        else:
            raise ValueError(
                f"Unsupported routing algorithm for {self.routing_algorithm}."
            )

    def __repr__(self):
        tab = "\t"
        line = "\n"
        res = f"{self.__class__.__name__}({line}"
        res += (
            f"{tab}(0): CapsuleLinear({self.input_dim}, {self.capsule_dimension}){line}"
        )
        res += f"{tab}(1): Routing(routing_steps={self.routing_steps}){line})"
        return res

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Unsqueeze to (batch_size, 1, input_caps, input_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)

        # W @ x = (1, num_caps, input_caps, caps_dim, input_dim) @ (batch_size, 1, input_caps, input_dim, 1) = (batch_size, num_caps, input_caps, caps_dim, 1)
        u_hat = torch.matmul(self.W, x)
        # Remove last dimension (batch_size, num_caps, input_caps, caps_dim)
        u_hat = u_hat.squeeze(-1)
        # Prevent gradient from flowing during routing iterations
        detached_u_hat = u_hat.detach()

        """
        Procedure 1 Routing algorithm.  
        1: procedure ROUTING(u^j|i, r, l)
        2:      for all capsule i in layer l and capsule j in layer (l + 1): bij ← 0. 
        3:      for r iterations do 
        4:          for all capsule i in layer l: ci ← softmax(bi) . softmax computes Eq. 3 
        5:          for all capsule j in layer (l + 1): sj ← ∑  i cij u^j|i  
        6:          for all capsule j in layer (l + 1): vj ← squash(sj) . squash computes Eq. 1 
        7:          for all capsule i in layer l and capsule j in layer (l + 1): bij ← bij + u^j|i.vj  
                return vj
        """

        b = torch.zeros(batch_size, self.num_class_caps, self.num_input_caps, 1).to(
            device
        )
        for r in range(self.routing_steps - 1):
            c = self.routing_activation(b)
            s = (c * detached_u_hat).sum(dim=2)
            v = squash(s)
            b += torch.matmul(detached_u_hat, v.unsqueeze(-1))

        # Perform last iteration on attached u_hat
        c = self.routing_activation(b)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)

        # return s  # We return the unsquashed vector to avoid problems with logits
        return squash(s)

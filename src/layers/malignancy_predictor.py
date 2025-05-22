import torch
import torch.nn as nn


class MalignancyPredictor(nn.Module):
    def __init__(self, num_attributes, capsule_dim, output_dim):
        """
        Args:
            num_attributes (int): Number of visual attributes.
            capsule_dim (int): Dimension of the capsule output (i.e. length of the capsule vector).
            output_dim (int): Dimension of the output (e.g., number of malignancy classes).
        """
        super(MalignancyPredictor, self).__init__()
        self.num_attributes = num_attributes
        self.capsule_dim = capsule_dim
        self.output_dim = output_dim

        # Define the layers of the malignancy predictor
        self.fc = nn.Linear(num_attributes * capsule_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)

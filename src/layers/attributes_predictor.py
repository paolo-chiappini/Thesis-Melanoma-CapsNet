import torch.nn as nn


class AttributesPredictor(nn.Module):
    def __init__(self, capsule_pose_dim, hidden_dim=32):
        super(AttributesPredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(capsule_pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # attribute logit
        )

    def forward(self, x):
        return self.classifier(x)

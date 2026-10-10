import torch
from torch import nn

# averages along the group dimension
class AvgGroupPool2d(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 5:
            return input.mean(dim=2)
        else:
            return input.mean(dim=1)

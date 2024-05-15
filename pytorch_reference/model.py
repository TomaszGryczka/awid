import torch
import torch.nn as nn
import torch.nn.functional as F
# Referencyjny model

class ReferenceModel(nn.Module):
    def __init__(self):
        super(ReferenceModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.d1 = nn.Linear(1014, 84)
        self.d2 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1014)
        x = F.relu(self.d1(x))
        x = self.d2(x)
        return F.log_softmax(x, dim=1)
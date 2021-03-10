import torch.nn as nn
import torch.nn.functional as F

#module records index of ReLU positive value at first time
class bw_relu(nn.Module):
    def __init__(self):
        super(bw_relu,self).__init__()

    def forward(self, x, relu_mask = None):
        if relu_mask is None:
            return F.relu(x)
        else:
            return x * relu_mask
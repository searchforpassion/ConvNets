import torch
import torch.nn as nn
from math import ceil

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
         
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))
 
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        groups=1,
        expand_ratio,
        reduction=4, #squeeze excitation
        survival_prob=0.8, # stochastic depth
        ):
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = ((in_channels == out_channels) and (stride == 1))
        hidden_dim = in_channels * expand_ratio
        # self.expand = (in_channels!=hidden_dim)
        reduced_dim = int(hidden_dim/reduction)
        
        if expand_ratio > 1:
            self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=1)
            
        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
            nn.BatchNorm2d(out_channels),
        )
        
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob)
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if expand_ratio > 1 else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        

class EfficientNet(nn.Module):
    pass 
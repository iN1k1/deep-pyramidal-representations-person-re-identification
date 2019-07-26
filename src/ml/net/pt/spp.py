import torch.nn as nn
import torch
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool', stripe=False):
        super(SpatialPyramidPooling, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type
        self.use_stripes = stripe

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_h = h // (2 ** i)
            kernel_w = w // (2 ** i)
            if self.use_stripes:
                kernel_w = w
            kernel_size = (kernel_h, kernel_w)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x
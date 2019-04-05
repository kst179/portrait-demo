import torch
from torch import nn
from torch.functional import F


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None):
        super(SelfAttentionBlock, self).__init__()

        self.key_channels = key_channels
        self.value_channels = value_channels

        if out_channels is None:
            out_channels = in_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1),
            nn.BatchNorm2d(self.key_channels),  # replaced InPlaceABN with usual BatchNorm
                                                # because I have no idea what that ABN doing...
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels, value_channels, kernel_size=1)

        self.f_out = nn.Conv2d(value_channels, out_channels, kernel_size=1)

        nn.init.constant_(self.f_out.weight.data, 0)  # don't know why but they use 0 init
        nn.init.constant_(self.f_out.bias.data, 0)  # in OCNet article code

    def forward(self, x):
        # input - [B x iC x H x W]
        batch_size, _, height, width = x.size()

        # value, query, key^T - [B x HW x (v/k/q)C], qC = kC
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        # sim_map - [B x HW x HW]
        sim_map = torch.matmul(query, key)
        sim_map.div_(self.key_channels ** 0.5)
        sim_map = F.softmax(sim_map, dim=-1)

        # context - [B x HW x vC]
        context = torch.matmul(sim_map, value)

        # back to [B x vC x H x W]
        context = context.permute(0, 2, 1).contiguous().view(batch_size, self.value_channels, height, width)

        return self.f_out(context)

import math
import einops
import torch.nn as nn
import torch


class SimpleUNet(nn.Module):

    def __init__(
            self, time_emb_dim, device) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim, device),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()

        )

        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        self.init_conv = nn.Conv2d(3, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])

        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        self.final_conv = nn.Conv2d(up_channels[-1], 3, 1)

    def forward(self, x, t):

        t = self.time_mlp(t)

        x = self.init_conv(x)

        intermediates = []

        for down in self.downs:
            x = down(x, t)
            intermediates.append(x)

        for up in self.ups:
            intermediate = intermediates.pop()
            x = torch.cat((x, intermediate), dim=1)
            x = up(x, t)

        return self.final_conv(x)


class Upsample(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):

        return self.upsample(x)


class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv_down = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):

        x = einops.rearrange(
            x,
            'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2
        )
        return self.conv_down(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv_1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn_0 = nn.BatchNorm2d(out_channels)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if up:

            self.transform = Upsample(out_channels)
            self.conv_0 = nn.Conv2d(
                2 * in_channels, out_channels, 3, padding=1)

        else:

            self.transform = Downsample(out_channels)
            self.conv_0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, t):

        x = self.bn_0(self.relu(self.conv_0(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = einops.rearrange(time_emb, 'b c -> b c 1 1')

        x = x + time_emb

        x = self.bn_1(self.relu(self.conv_1(x)))

        return self.transform(x)

import math
import einops
import torch.nn as nn
import torch
import torchvision.transforms as transforms


class UNet(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_emb_dim, device) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim, device),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()

        )

        self.contracting_block_0 = ContractingBlock(
            in_channels, 64, time_emb_dim)
        self.contracting_block_1 = ContractingBlock(64, 128, time_emb_dim)
        self.contracting_block_2 = ContractingBlock(128, 256, time_emb_dim)
        self.contracting_block_3 = ContractingBlock(256, 512, time_emb_dim)

        self.head = Head(512, 1024)

        self.expansive_block_0 = ExpansiveBlock(1024, 512, time_emb_dim)
        self.expansive_block_1 = ExpansiveBlock(512, 256, time_emb_dim)
        self.expansive_block_2 = ExpansiveBlock(256, 128, time_emb_dim)
        self.expansive_block_3 = ExpansiveBlock(128, 64, time_emb_dim)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):

        t = self.time_mlp(t)

        x, intermediate_0 = self.contracting_block_0(x, t)
        x, intermediate_1 = self.contracting_block_1(x, t)
        x, intermediate_2 = self.contracting_block_2(x, t)
        x, intermediate_3 = self.contracting_block_3(x, t)

        x = self.head(x)

        x = self.expansive_block_0(x, t, intermediate_3)
        x = self.expansive_block_1(x, t, intermediate_2)
        x = self.expansive_block_2(x, t, intermediate_1)
        x = self.expansive_block_3(x, t, intermediate_0)

        x = self.final_conv(x)
        x = nn.functional.interpolate(x, (448, 576))

        return x


class ContractingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim) -> None:
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)

        self.silu = nn.SiLU()
        self.group_nom = nn.GroupNorm(1, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t):

        x = self.group_nom(self.silu(self.conv0(x)))

        time_emb = self.time_mlp(t)
        time_emb = einops.rearrange(time_emb, 'b c -> b c 1 1')
        x = x + time_emb

        intermediate = self.group_nom(self.silu(self.conv1(x)))
        x = self.pool(intermediate)

        return x, intermediate


class Head(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))

        return x


class ExpansiveBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim):

        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv_up = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)

        self.silu = nn.SiLU()
        self.group_norm = nn.GroupNorm(1, out_channels)

    def forward(self, x, t, intermediate):

        x = self.upsample(x)
        x = self.conv_up(x)

        intermediate = transforms.functional.center_crop(
            intermediate, [x.shape[2], x.shape[3]])

        x = torch.cat((x, intermediate), dim=1)

        x = self.group_norm(self.silu(self.conv0(x)))

        time_emb = self.time_mlp(t)
        time_emb = einops.rearrange(time_emb, 'b c -> b c 1 1')
        x = x + time_emb

        x = self.group_norm(self.silu(self.conv1(x)))

        return x


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

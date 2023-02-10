import torch.nn as nn
import einops
import torch
import math
from functools import partial


class Unet(nn.Module):

    def __init__(self, dim, device, dim_mults=(1, 2, 4, 8), in_channels=3,
                 out_channels=3, resnet_block_groups=4):

        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, dim, 1, padding=0)

        dims = [dim] + [dim * mult for mult in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_emb_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, device),
            nn.Linear(dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(in_out):

            if i >= len(in_out) - 1:
                is_last = True
            else:
                is_last = False

            self.downs.append(nn.ModuleList([
                ResnetBlock(
                    dim_in, dim_in, time_emb_dim, resnet_block_groups),
                ResnetBlock(
                    dim_in, dim_in, time_emb_dim, resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        self.head1 = ResnetBlock(
            dims[-1], dims[-1], time_emb_dim, resnet_block_groups)
        self.head_attention = Residual(PreNorm(dims[-1], Attention(dims[-1])))
        self.head2 = ResnetBlock(
            dims[-1], dims[-1], time_emb_dim, resnet_block_groups)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):

            if i >= len(in_out) - 1:
                is_last = True
            else:
                is_last = False

            self.ups.append(nn.ModuleList([
                ResnetBlock(
                    dim_out + dim_in, dim_out,
                    time_emb_dim,
                    resnet_block_groups
                ),
                ResnetBlock(
                    dim_out + dim_in, dim_out,
                    time_emb_dim,
                    resnet_block_groups
                ),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_block = ResnetBlock(
            dim * 2, dim, time_emb_dim, resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t):

        x = self.init_conv(x)
        residual = x.clone()

        t = self.time_mlp(t)

        h = []

        for block1, block2, attention, downsample in self.downs:

            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attention(x)
            h.append(x)

            x = downsample(x)

        x = self.head1(x, t)
        x = self.head_attention(x)
        x = self.head2(x, t)

        for block1, block2, attention, upsample in self.ups:

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attention(x)

            x = upsample(x)

        x = torch.cat((x, residual), dim=1)

        x = self.final_block(x, t)
        return self.final_conv(x)


class WeightStandardizedCOnv2d(nn.Conv2d):

    """siehe https://arxiv.org/abs/1903.10520"""

    def forward(self, x):

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = einops.reduce(
            self.weight,
            "o ... -> o 1 1 1", "mean"
        )
        var = einops.reduce(
            self.weight,
            "o ... -> o 1 1 1", partial(torch.var, unbiased=False)
        )

        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()

        return torch.nn.functional.conv2d(
            x, normalized_weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups
        )


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = WeightStandardizedCOnv2d(
            in_channels, out_channels, 3, padding=1)
        self.group_norm = nn.GroupNorm(groups, out_channels)
        self.silu = nn.SiLU()

    def forward(self, x, scale_shift=None):

        x = self.group_norm(self.conv(x))

        if scale_shift is not None:

            x = x * (scale_shift[0] + 1)
            x += scale_shift[1]

        x = self.silu(x)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):

        super().__init__()

        # maps on time_emb_dim for each scale and shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb):

        time_emb = self.mlp(time_emb)
        time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(
            out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_down = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x):

        x = einops.rearrange(
            x,
            'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2
        )
        return self.conv_down(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):

        return self.upsample(x)


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

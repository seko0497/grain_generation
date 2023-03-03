import torch.nn as nn
import einops
import torch
import math
from functools import partial


class Unet(nn.Module):

    def __init__(self, dim, device, dim_mults=(1, 2, 4, 8), in_channels=3,
                 out_channels=3, resnet_block_groups=32, num_resnet_blocks=2,
                 attention_dims=(32, 16, 8), dropout=0.0, spade=False,
                 num_classes=None):

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

        self.label_dist_embedding = nn.Linear(num_classes, time_emb_dim)

        self.attention_dims = attention_dims

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(in_out):

            if i >= len(in_out) - 1:
                is_last = True
            else:
                is_last = False

            self.downs.append(nn.ModuleList([
                nn.ModuleList([ResnetBlock(
                    dim_in, dim_in, time_emb_dim, resnet_block_groups,
                    dropout=dropout)
                    for _ in range(num_resnet_blocks)]),
                Residual(PreNorm(dim_in, Attention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        self.head1 = ResnetBlock(
            dims[-1], dims[-1], time_emb_dim, resnet_block_groups,
            dropout=dropout, spade=spade, num_classes=num_classes)
        self.head_attention = Residual(PreNorm(dims[-1], Attention(dims[-1])))
        self.head2 = ResnetBlock(
            dims[-1], dims[-1], time_emb_dim, resnet_block_groups,
            dropout=dropout, spade=spade, num_classes=num_classes)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):

            if i >= len(in_out) - 1:
                is_last = True
            else:
                is_last = False

            self.ups.append(nn.ModuleList([
                nn.ModuleList([ResnetBlock(
                    dim_out + dim_in, dim_out,
                    time_emb_dim,
                    resnet_block_groups,
                    dropout=dropout,
                    spade=spade, num_classes=num_classes
                )for _ in range(num_resnet_blocks)]),
                Residual(PreNorm(dim_out, Attention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_block = ResnetBlock(
            dim * 2, dim, time_emb_dim, resnet_block_groups, dropout=dropout,
            spade=spade, num_classes=num_classes)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t, mask=None, label_dist=None):

        x = self.init_conv(x)
        residual = x.clone()

        t = self.time_mlp(t)
        if label_dist is not None:
            t = t + self.label_dist_embedding(label_dist)

        h = []
        ds = 1

        for blocks, attention, downsample in self.downs:

            for block in blocks[:-1]:

                x = block(x, t)
                h.append(x)

            x = blocks[-1](x, t)
            if ds in self.attention_dims:
                x = attention(x)
            h.append(x)

            x = downsample(x)
            ds *= 2

        x = self.head1(x, t, mask=mask)
        x = self.head_attention(x)
        x = self.head2(x, t, mask=mask)

        for blocks, attention, upsample in self.ups:

            for block in blocks[:-1]:
                x = torch.cat((x, h.pop()), dim=1)
                x = block(x, t, mask=mask)

            x = torch.cat((x, h.pop()), dim=1)
            x = blocks[-1](x, t, mask=mask)
            if ds in self.attention_dims:
                x = attention(x)

            x = upsample(x)
            ds //= 2

        x = torch.cat((x, residual), dim=1)

        x = self.final_block(x, t, mask=mask)
        return self.final_conv(x)


class SPADE(nn.Module):

    def __init__(self, in_channels, num_classes, groups=8):

        super().__init__()

        self.group_norm = nn.GroupNorm(groups, in_channels, affine=False)
        hiddem_dim = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(num_classes, hiddem_dim, 3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(hiddem_dim, in_channels, 3, padding=1)
        self.mlp_beta = nn.Conv2d(hiddem_dim, in_channels, 3, padding=1)

    def forward(self, x, mask):

        x = self.group_norm(x)

        mask = nn.functional.interpolate(
            mask, size=x.shape[2:], mode="nearest")
        mask = self.mlp_shared(mask)
        gamma = self.mlp_gamma(mask)
        beta = self.mlp_beta(mask)

        return x * (1 + gamma) + beta


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, groups=8, dropout=None,
                 spade=False, num_classes=None):
        super().__init__()

        if spade:
            self.spade = SPADE(in_channels, num_classes, groups)
        else:
            self.group_norm = nn.GroupNorm(groups, in_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout else None

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, scale_shift=None, mask=None):

        if mask is not None:
            assert hasattr(self, "spade"), ("if mask is given, "
                                            "model must have SPADE")
            x = self.spade(x, mask)
        else:
            x = self.group_norm(x)

        if scale_shift:
            x = x * (scale_shift[0] + 1)
            x += scale_shift[1]

        x = self.silu(x)

        if self.dropout:
            x = self.dropout(x)

        return self.conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8,
                 dropout=0.0, spade=False, num_classes=None):

        super().__init__()

        # maps on time_emb_dim for each scale and shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.dropout = dropout

        self.block1 = Block(in_channels, out_channels, groups=groups,
                            spade=spade, num_classes=num_classes)
        self.block2 = Block(out_channels, out_channels, groups=groups,
                            dropout=dropout, spade=spade,
                            num_classes=num_classes)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb, mask=None):

        time_emb = self.mlp(time_emb)
        time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, mask=mask)
        h = self.block2(h, scale_shift, mask=mask)

        return h + self.res_conv(x)


class QKVAttention(nn.Module):

    # ported from
    # ("https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py")"

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class Attention(nn.Module):

    # ported from
    # ("https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py")"

    def __init__(self, channels, num_heads=4, num_head_channels=32, groups=8):

        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), (f"q,k,v channels {channels} is not divisible by"
                f"num_head_channels {num_head_channels}")

            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


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


# x = torch.empty((4, 3, 64, 64))

# model = Unet(128, torch.device("cpu"), dim_mults=[1, 1, 2, 2, 4, 4],
#              cond=True, num_classes=3)

# map = torch.zeros((4, 3, 64, 64))

# t = torch.full((4, ), 42)
# model(x, t)

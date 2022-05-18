# This code is taken from how-vit-works repo:
### Attention Blocks defined below:
import torch
from torch import nn, einsum
from functools import partial
from itertools import cycle
from einops import rearrange
import numpy as np

import matplotlib.pyplot as plt

# The actual self-attetion mechnanism
class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn

# The local attention applied to each patch
class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)
        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d


#######################################
# This is the main inherited MSA!!!!!!#
#######################################


class AttentionBlockA(nn.Module):
    expansion = 4

    def __init__(self, dim_in, window_size, heads, dim_out=None, *,
                 dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion))
            self.shortcut.append(norm(dim_out * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv = nn.Sequential(
            conv1x1(dim_in, width, stride=stride),
            norm(width),
            activation(),
        )
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = norm(dim_out * self.expansion)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()
        
    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv(x)
        x, attn = self.attn(x)
        x = self.norm(x)
        x = self.sd(x) + skip

        if not self.training:
            self.attn_output = attn

        return x

# Create MSA:
class AttentionBasicBlockA(AttentionBlockA):
    expansion = 1

def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1)

def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)


# I don't know what this does:

class DropPath(nn.Module):
    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
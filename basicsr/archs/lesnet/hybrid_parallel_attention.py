#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OSA.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:07:42 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from basicsr.archs.lesnet.layernorm import LayerNorm2d


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor ** 2),
                     kernel_size,
                     padding=1,
                     bias=bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


import torch.nn as nn


def pixelunshuffle_block(in_channels,
                         out_channels,
                         downscale_factor=2,
                         kernel_size=3,
                         bias=False):
    """
    Downsample features according to `downscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels * (downscale_factor ** 2),
                     out_channels,
                     kernel_size,
                     padding=1,
                     bias=bias)
    pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
    return nn.Sequential(*[pixel_unshuffle, conv])


# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1, 0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1, 1, 0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


# window and global spatial attention related classes(channel//2)
class hybrid_Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim // 2, dim // 2, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

        sr_ratio = 16  # aggregation rate
        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.sr = nn.Conv2d(dim, dim, kernel_size=3, stride=sr_ratio, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.kv1 = nn.Linear(dim, dim, bias=True)
        self.q1 = nn.Linear(dim, dim // 2, bias=True)
        self.q2 = nn.Linear(dim, dim // 2, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        # self.conv_down = nn.Conv2d(dim // 2, dim // 2, kernel_size=8, stride=8)
        ####lesnet8,9
        self.conv_down = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=8, padding=1)
        ####lesnet9 multi-pixelshuffle_block to save params
        # self.conv_up = nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=8, stride=8)
        self.conv_up = nn.Sequential(
                nn.Upsample(scale_factor=8),
                nn.Conv2d(dim//2, dim // 2, kernel_size=3, stride=1, padding=1),
                )
        self.attn_drop1 = nn.Dropout(dropout)


    def forward(self, x):
        x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')
        input = x
        B, C, H, W = input.shape
        window_size = 8
        x = self.q1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)  # w1,w2: window_size
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')  # B*window_num*window_num, window_size*window_size, C

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h),
                      (q, k, v))  # B*window_num*window_num, head, window_size*window_size, C//h

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        out = rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)
        # out = rearrange(out, 'b x y w1 w2 d -> b d (x w1) (y w2)')

        up_down_scale = 8
        # global
        input_q = self.q2(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input_q = self.conv_down(input_q)
        q1 = input_q.reshape(B, H * W // up_down_scale // up_down_scale, self.heads // 2, C // self.heads).permute(0, 2,
                                                                                                                   1,
                                                                                                                   3)  # B,H*W,head//2,C//head
        x_ = input  # B, C, H, W
        x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 降采样
        x_1 = self.act(self.norm(x_1))
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.heads // 2, C // self.heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # B head Nq Nkv
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v1).transpose(2, 3).reshape(B, C // 2, H // up_down_scale, W // up_down_scale)
        out1 = self.conv_up(out1)
        out1 = self.attn_drop1(out1)
        out1 = rearrange(out1, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)
        output = torch.cat([out, out1], dim=5)

        return output


# window and global spatial attention related classes(channel//2), modified the dropout progress
class hybrid_Attention_s(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim//2, dim//2 * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim // 2, dim // 2, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

        sr_ratio = 16  # aggregation rate
        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.sr = nn.Conv2d(dim, dim, kernel_size=3, stride=sr_ratio, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.kv1 = nn.Linear(dim, dim, bias=True)
        self.q1 = nn.Linear(dim, dim // 2, bias=True)
        self.q2 = nn.Linear(dim, dim // 2, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        ####lesnet8,9
        self.conv_down = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=8, padding=1)
        ####lesnet9 multi-pixelshuffle_block to save params
        # self.conv_up = nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=8, stride=8)
        self.conv_up = nn.Sequential(
                nn.Upsample(scale_factor=8),
                nn.Conv2d(dim//2, dim // 2, kernel_size=3, stride=1, padding=1),
                )
        self.to_output = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')
        input = x
        B, C, H, W = input.shape
        window_size = 8

        x = self.q1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)  # w1,w2: window_size
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')  # B*window_num*window_num, window_size*window_size, C

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h),
                      (q, k, v))  # B*window_num*window_num, head, window_size*window_size, C//h

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out
        # out = self.to_out(out)
        out = rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)

        # global
        up_down_scale = 8
        # global
        input_q = self.q2(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input_q = self.conv_down(input_q)
        q1 = input_q.reshape(B, H * W // up_down_scale // up_down_scale, self.heads // 2, C // self.heads).permute(0, 2,
                                                                                                                   1,
                                                                                                                   3)  # B,H*W,head//2,C//head
        x_ = input  # B, C, H, W
        x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 降采样
        x_1 = self.act(self.norm(x_1))
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.heads // 2, C // self.heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # B head Nq Nkv
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v1).transpose(2, 3).reshape(B, C // 2, H // up_down_scale, W // up_down_scale)
        out1 = self.conv_up(out1)
        out1 = rearrange(out1, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)
        output = torch.cat([out, out1], dim=5)
        output = self.to_output(output)

        return output


# window and global spatial attention related classes(local attention--no channel//2)
class hybrid_Attention_ss(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Linear(dim, dim // 2, bias=False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim // 2, bias=False),
        #     nn.Dropout(dropout)
        # )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

        sr_ratio = 16  # aggregation rate
        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.sr = nn.Conv2d(dim, dim, kernel_size=3, stride=sr_ratio, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.kv1 = nn.Linear(dim, dim, bias=True)
        self.q1 = nn.Linear(dim, dim // 2, bias=True)
        self.q2 = nn.Linear(dim, dim // 2, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        ####lesnet8,9
        self.conv_down = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=8, padding=1)
        ####lesnet9 multi-pixelshuffle_block to save params
        # self.conv_up = nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=8, stride=8)
        self.conv_up = nn.Sequential(
                nn.Upsample(scale_factor=8),
                nn.Conv2d(dim//2, dim // 2, kernel_size=3, stride=1, padding=1),
                )
        self.to_output = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')
        input = x
        B, C, H, W = input.shape
        window_size = 8

        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)  # w1,w2: window_size
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')  # B*window_num*window_num, window_size*window_size, C

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h),
                      (q, k, v))  # B*window_num*window_num, head, window_size*window_size, C//h

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        out = rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)
        # out = rearrange(out, 'b x y w1 w2 d -> b d (x w1) (y w2)')

        # global
        up_down_scale = 8
        # global
        input_q = self.q2(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input_q = self.conv_down(input_q)
        q1 = input_q.reshape(B, H * W // up_down_scale // up_down_scale, self.heads // 2, C // self.heads).permute(0, 2,
                                                                                                                   1,
                                                                                                                   3)  # B,H*W,head//2,C//head
        x_ = input  # B, C, H, W
        x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 降采样
        x_1 = self.act(self.norm(x_1))
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.heads // 2, C // self.heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # B head Nq Nkv
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v1).transpose(2, 3).reshape(B, C // 2, H // up_down_scale, W // up_down_scale)
        out1 = self.conv_up(out1)
        out1 = rearrange(out1, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=window_size, w2=window_size)
        output = torch.cat([out, out1], dim=5)
        output = self.to_output(output)

        return output


class Block_Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            bias=False,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.ps = window_size
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # project for queries, keys, values
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h=self.heads, w1=self.ps,
                                          w2=self.ps), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, '(b x y) head (w1 w2) d -> b (head d) (x w1) (y w2)', x=h // self.ps, y=w // self.ps,
                        head=self.heads, w1=self.ps, w2=self.ps)

        out = self.to_out(out)
        return out


###___Channel_Attention类通过将输入张量的高和宽分别划分为若干个小块，并将每个小块的像素值视为通道，从而将输入张量重排为一个二维的矩阵，然后对该矩阵进行通道注意力计算。

class Channel_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps,
                                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h // self.ps, w=w // self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


###___Channel_Attention_grid类则是直接将输入张量重排为一个二维的矩阵，其中每个像素点都被视为一个通道，然后对该矩阵进行通道注意力计算。

class Channel_Attention_grid(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)', ph=self.ps, pw=self.ps,
                                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)', h=h // self.ps, w=w // self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(

            #####____ Local Convolution Block (LCB)
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25
            ),

            #####____ block-like attention___Meso-OSA
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),  # spatial-attention
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # channel-attention
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            #####____ grid-like attention___Global-OSA
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
                                                                     window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class OSA_Block_modified(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block_modified, self).__init__()

        w = window_size

        self.norm = LayerNorm2d(channel_num)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel_num * 2, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.layer1 = nn.Sequential(
            #####____ Local Convolution Block (LCB)
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25)
        )

        self.layer = nn.Sequential(

            #####____ Local Convolution Block (LCB)
            # MBConv(
            #     channel_num,
            #     channel_num,
            #     downsample = False,
            #     expansion_rate = 1,
            #     shrinkage_rate = 0.25
            # ),

            #####____ block-like attention___Meso-OSA
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),  # spatial-attention
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # channel-attention
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            #####____ grid-like attention___Global-OSA
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
                                                                     window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        x = self.norm(self.conv1(x))

        out = self.layer(x)
        out1 = self.layer1(x)

        out = self.conv2(torch.cat([out, out1], 1)) + x
        return out


class OSA_Block_hybrid_modified(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block_hybrid_modified, self).__init__()

        w = window_size

        self.norm = LayerNorm2d(channel_num)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel_num * 2, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.layer1 = nn.Sequential(
            #####____ Local Convolution Block (LCB)
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25)
        )

        self.layer = nn.Sequential(

            #####____ Local Convolution Block (LCB)
            # MBConv(
            #     channel_num,
            #     channel_num,
            #     downsample = False,
            #     expansion_rate = 1,
            #     shrinkage_rate = 0.25
            # ),

            #####____ block-like attention___Meso-OSA
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排 b:B,d:C,w:W,
            PreNormResidual(channel_num, hybrid_Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                          window_size=window_size, with_pe=with_pe)),
            # spatial-attention
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # channel-attention
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            #####____ grid-like attention___Global-OSA
            # Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            # PreNormResidual(channel_num, hybrid_Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
            #                                               window_size=window_size, with_pe=with_pe)),
            # Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            #
            # # channel-like attention
            # Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
            #                                                          window_size=window_size)),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        x = self.norm(self.conv1(x))

        out = self.layer(x)
        out1 = self.layer1(x)

        out = self.conv2(torch.cat([out, out1], 1)) + x
        return out


class OSA_Block_hybrid_modified_s(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block_hybrid_modified_s, self).__init__()

        w = window_size

        self.norm = LayerNorm2d(channel_num)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel_num * 2, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.layer1 = nn.Sequential(
            #####____ Local Convolution Block (LCB)
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25)
        )

        self.layer = nn.Sequential(

            #####____ Local Convolution Block (LCB)
            # MBConv(
            #     channel_num,
            #     channel_num,
            #     downsample = False,
            #     expansion_rate = 1,
            #     shrinkage_rate = 0.25
            # ),

            #####____ block-like attention___Meso-OSA
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排 b:B,d:C,w:W,
            PreNormResidual(channel_num,
                            hybrid_Attention_s(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                      window_size=window_size, with_pe=with_pe)),  # spatial-attention
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # channel-attention
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            #####____ grid-like attention___Global-OSA
            # Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            # PreNormResidual(channel_num,
            #                 hybrid_Attention_s(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
            #                                           window_size=window_size, with_pe=with_pe)),
            # Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            #
            # # channel-like attention
            # Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
            #                                                          window_size=window_size)),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            #####____ block-like attention___Meso-OSA   channel-to-spatial
            # Conv_PreNormResidual(channel_num,
            #                      Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # # channel-attention
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # # FeedForward network
            #
            # # ____ block-like attention
            # Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排 b:B,d:C,w:W,
            # PreNormResidual(channel_num,
            #                 hybrid_Attention_s(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
            #                                           window_size=window_size, with_pe=with_pe)),  # spatial-attention
            # Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network
        )

    def forward(self, x):
        x = self.norm(self.conv1(x))

        out = self.layer(x)
        out1 = self.layer1(x)

        out = self.conv2(torch.cat([out, out1], 1)) + x
        return out



class OSA_Block_hybrid_modified_m(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block_hybrid_modified_m, self).__init__()

        w = window_size

        self.norm = LayerNorm2d(channel_num)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(channel_num * 2, channel_num, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.layer1 = nn.Sequential(
            #####____ Local Convolution Block (LCB)
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25)
        )

        self.layer = nn.Sequential(

            #####____ Local Convolution Block (LCB)
            # MBConv(
            #     channel_num,
            #     channel_num,
            #     downsample = False,
            #     expansion_rate = 1,
            #     shrinkage_rate = 0.25
            # ),

            #####____ block-like attention___Meso-OSA
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # 对输入进行维度重排 b:B,d:C,w:W,
            PreNormResidual(channel_num,
                            hybrid_Attention_ss(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                      window_size=window_size, with_pe=with_pe)),  # spatial-attention
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            # channel-attention
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            # FeedForward network

            # #####____ grid-like attention___Global-OSA
            # Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            # PreNormResidual(channel_num,
            #                 hybrid_Attention_ss(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
            #                                           window_size=window_size, with_pe=with_pe)),
            # Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            #
            # # channel-like attention
            # Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
            #                                                          window_size=window_size)),
            # Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        x = self.norm(self.conv1(x))

        out = self.layer(x)
        out1 = self.layer1(x)

        out = self.conv2(torch.cat([out, out1], 1)) + x
        return out


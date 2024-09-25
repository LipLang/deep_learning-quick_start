"""
N-HiTS

Refs:
    - [paper](https://arxiv.org/abs/2201.12886)
    - [darts](https://github.com/unit8co/darts/blob/master/darts/models/forecasting/nhits.py)
"""

import torch
import numpy as np
from torch import nn, optim

ACTIVATIONS = ["ReLU", "RReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "GELU",]
POOLS = ["MaxPool1d", "AvgPool1d",]
NORMS = ["BatchNorm1d", "LayerNorm",]


#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>


class NhitsBlock(nn.Module):

    def __init__(self, input_len, output_len, leadingMLP_hidden_dims, pool_kernel_dim, downsample_factor,
                 dropout_rate=0.1, activation='ReLu', pool='MaxPool1d', norm='BatchNorm1d'):

        super().__init__()
        #
        def get_nn_attr(attr, checklist=[]):
            if not attr: # attr in {'', None, [], 0, ...}
                return nn.Identity
            else:
                if checklist: assert attr in checklist, f"'{attr}' is not in {checklist}"
                return getattr(nn, attr)

        activation    = get_nn_attr(activation, ACTIVATIONS)
        pooling       = get_nn_attr(pool, POOLS)
        normalization = get_nn_attr(norm, NORMS)
        dropout       = nn.Dropout if dropout_rate>0 else nn.Identity
        #
        self.input_len  = input_len
        self.output_len = output_len

        # entry pooling layer, 迫使Nhits块专注于大尺寸(低频)内容(文中称多速率信号采样: 不同的块对应不同的输入信号采样频率)
        self.pooling_layer = pooling(kernel_size=pool_kernel_dim, stride=pool_kernel_dim, padding=pool_kernel_dim // 2)

        # leading linear layers (MLP stack)
        if not hasattr(leadingMLP_hidden_dims, '__iter__'): leadingMLP_hidden_dims = [leadingMLP_hidden_dims] # 是个标量(容错)

        leadingMLP_dims = [np.ceil(self.input_len/pool_kernel_dim + 1)] + leadingMLP_hidden_dims
        leadingMLP_list = [
            [nn.Linear(leadingMLP_dims[i-1], leadingMLP_dims[i]), activation(), dropout()]
            for i in range(1, len(leadingMLP_dims))
        ]
        self.leading_MLP = nn.Sequential(*[lyr for mlp in leadingMLP_list for lyr in mlp])

        # forecast and backcast linear mapping (theta_l^f, theta_l^b)
        self.backcast_linear = nn.Linear(leadingMLP_dims[-1], max( input_len//downsample_factor, 1))
        self.forecast_linear = nn.Linear(leadingMLP_dims[-1], max(output_len//downsample_factor, 1))

    def forward(self, x):
        # 1. 先进行一步池化
        # [batch_size, seq_len]->[batch_size, 1, seq_len]->Pool(...)->[batch_size, 1, new_seq_len]->[batch_size, new_seq_len]
        x = x.unsqueeze(1) # 在张量的第二个维度(索引1)添加一个维度. 通常将2D转换为3D, 添加代表通道(channel)的维度
        x = self.pooling_layer(x)  # 一维池化层的期望输入是3D的, 格式为 `[batch_size, channels, length]`
        x = x.squeeze(1) # 移除张量的第二个维度(如果该维度的大小为 1)

        # 2. leading linear layers (MLP stack)
        x = self.leading_MLP(x)

        # 3. theta
        theta_backcast = self.backcast_linear(x)
        theta_forecast = self.forecast_linear(x)

        # 4. 对结果进行插值，恢复到原始尺寸
        theta_backcast = theta_backcast.unsqueeze(1) # to shape [batch_size, "channel_num", time]
        interpolated_backcast = nn.functional.interpolate(
            pooled_backcast, size=self.input_len, mode='linear', align_corners=False
        )
        interpolated_backcast = interpolated_backcast.squeeze(1) # del the 2nd dim: "channel_num"

        theta_forecast = theta_forecast.unsqueeze(1)
        interpolated_forecast = nn.functional.interpolate(
            pooled_forecast, size=self.output_len, mode='linear', align_corners=False
        )
        interpolated_forecast = interpolated_forecast.squeeze(1)

        return interpolated_backcast, interpolated_forecast


#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>


class NhitsStack(nn.Module):
    # cast_layer_num, cast_hidden_size, pool_kernel_dims
    # leadingMLP_hidden_dims, pool_kernel_dim, downsample_factor
    def __init__(self, input_len, output_len, block_num, leadingMLP_hidden_dims, pool_kernel_dim, downsample_factor,
                 dropout_rate=0.1, activation='ReLu', pool='MaxPool1d', norm='BatchNorm1d'):

        super().__init__()
        #
        self.input_len  = input_len
        self.output_len = output_len

        self.blocks = nn.ModuleList([
            NhitsBlock(input_len, output_len, leadingMLP_hidden_dims, pool_kernel_dim,
                       downsample_factor, dropout_rate, activation, pool, norm,
            ) for _ in range(block_num)
        ])

    def forward(self, x):
        forecast = torch.zeros(x.size(0), self.output_len, device=x.device, dtype=x.dtype,)

        residual = x # stack-level residual
        for block in self.blocks:
            block_backcast, block_forecast = block(residual)
            residual = residual - block_backcast
            forecast = forecast + block_forecast

        return residual, forecast


#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>


class NhitsModel(nn.Module):
    def __init__(self, input_len, output_len, input_dim, output_dim, block_num_per_stack,
                 block_leadingMLP_hidden_dims, block_pool_kernel_dim, block_downsample_factor,
                 dropout_rate=0.1, activation='ReLu', pool='MaxPool1d', norm='BatchNorm1d'):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim  # target and the covariates
        self.output_dim = output_dim
        self.input_len_multi = self.input_len * self.input_dim
        self.output_len_multi = self.output_len * self.output_dim

        self.stacks = nn.ModuleList([
            NhitsStack(
                input_len_multi, output_len_multi, block_num_per_stack, block_leadingMLP_hidden_dims, block_pool_kernel_dim,
                block_downsample_factor, dropout_rate, activation, pool, norm
            ) for _ in range(stack_num)
        ])

    def forward(self, x):
        forecast = torch.zeros(x.size(0), self.output_len_multi, device=x.device, dtype=x.dtype,)

        residual = x
        for stack in self.stacks:
            stack_backcast, stack_forecast = stack(residual)
            residual = residual - stack_backcast # 前xxx
            forecast = forecast + stack_forecast # xxx之后

        return residuals, forecast


"""
N-Beats

Ref: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/nbeats.py
"""

import enum, torch
import numpy as np
from typing import List, NewType
from torch import nn, optim
from einops import rearrange

EPSILON = 1e-7
ACTIVATIONS = ["ReLU", "RReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "GELU",]
POOLS = ["MaxPool1d", "AvgPool1d",]
NORMS = ["BatchNorm1d", "LayerNorm",]

def get_nn_attr(attr, checklist=[]):
    if not attr: # attr in {'', None, [], 0, ...}
        return nn.Identity
    else:
        if checklist: assert attr in checklist, f"'{attr}' is not in {checklist}"
        return getattr(nn, attr)


class GType(enum.Enum):
    GENERIC  = 0 # 水平
    TREND    = 1 # 趋势
    WAVEFORM = 2 # 季节(波动, 形态)


class TrendGenerator(nn.Module):
    def __init__(self, expansion_coef_dim, target_len):
        super().__init__()
        # basis is of size (expansion_coef_dim, target_len):
        basis = torch.stack([(torch.arange(target_len)/target_len)**i for i in range(expansion_coef_dim)], dim=1,).T
        # ^-- that is, 每个for i生成一个长为target_len的向量, 共生成expansion_coef_dim个
        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class WaveformGenerator(nn.Module):
    def __init__(self, target_len):
        super().__init__()
        half_len_sub1 = target_len//2 - 1
        cos_vectors = [torch.cos(2*np.pi*i * torch.arange(target_len)/target_len) for i in range(1, half_len_sub1+1)]
        sin_vectors = [torch.sin(2*np.pi*i * torch.arange(target_len)/target_len) for i in range(1, half_len_sub1+1)]
        # basis is of size (2 * (target_len//2 - 1) + 1, target_len):
        basis = torch.stack([torch.ones(target_len)] + cos_vectors + sin_vectors, dim=1).T
        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class NBEATSBlock(nn.Module):
    def __init__(
        self, layer_num: int, layer_width: int, expansion_coef_dim: int, input_chunk_len: int, target_len: int,
        g_type: GType, norm: str, dropout_rate: float, activation: str, **kwargs
    ):
        """
        The implemention of the basic building block of the N-BEATS architecture.
        The blocks produce outputs of size target_len; i.e. "one vector per parameter".

        Parameters
        ----------
        layer_num: The number of fully connected layers preceding the final forking layers.
        layer_width: The number of neurons that make up each fully connected layer.
        expansion_coef_dim: The dim of the waveform generator parameters, also known as expansion coefficients.
        input_chunk_len: The length of the input sequence fed to the model.
        target_len: The length of the forecast of the model.
        g_type: The type of function that is implemented by the waveform generator.
        norm: the norm function name
        dropout_rate: Dropout probability
        activation: The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_len)`: Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_len)`: Tensor containing the 'backcast' of the block,
              which represents an approximation of `x` given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_len)`: Tensor containing the forward forecast of the block.
        """
        super().__init__(**kwargs)

        self.layer_num = layer_num
        self.layer_width = layer_width
        self.target_len = target_len
        self.g_type = g_type
        self.dropout_rate = dropout_rate
        self.norm = get_nn_attr(norm)
        self.activation = get_nn_attr(activation)

        # fully connected stack before fork
        common_linear_body = [
            [nn.Linear(input_chunk_len, layer_width)]
        ] + [
            [
                nn.Linear(layer_width, layer_width),
                self.norm(num_features=self.layer_width),
                torch.Dropout(p=self.dropout_rate) if self.dropout_rate>0 else nn.Identity(),
            ] for _ in range(layer_num - 1)
        ]

        common_body_stack = nn.Sequential(*[lyr for mdlst in common_linear_body for lyr in mdlst]) # 原来是ModuleList
        self.common_body = nn.Sequential(common_body_stack, self.activation())

        # Fully connected layer producing forecast/backcast expansion coefficients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == GType.GENERIC:
            self.backcast_linear_head = nn.Linear(layer_width, expansion_coef_dim)
            self.forecast_linear_head = nn.Linear(layer_width, expansion_coef_dim)
            self.backcast_g = nn.Linear(expansion_coef_dim, input_chunk_len)
            self.forecast_g = nn.Linear(expansion_coef_dim, target_len)
        elif g_type == GType.TREND:
            self.backcast_linear_head = nn.Linear(layer_width, expansion_coef_dim)
            self.forecast_linear_head = nn.Linear(layer_width, expansion_coef_dim)
            self.backcast_g = TrendGenerator(expansion_coef_dim, input_chunk_len)
            self.forecast_g = TrendGenerator(expansion_coef_dim, target_len)
        elif g_type == GType.WAVEFORM:
            self.backcast_linear_head = nn.Linear(layer_width, 2 * (input_chunk_len // 2 - 1) + 1)
            self.forecast_linear_head = nn.Linear(layer_width, 2 * (target_len // 2 - 1) + 1)
            self.backcast_g = WaveformGenerator(input_chunk_len)
            self.forecast_g = WaveformGenerator(target_len)
        else:
            raise ValueError(f"g_type `{g_type}` is not supported")

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.common_body(x) # fully connected layer stack

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_head(x)
        theta_forecast = self.forecast_linear_head(x).view(batch_size, -1) # set the expansion coefs in last dim for forecasts

        # waveform generator applications (project the expansion coefs onto basis vectors)
        x_hat = self.backcast_g(theta_backcast)
        y_hat = self.forecast_g(theta_forecast).reshape(x.shape[0], self.target_len, 1) # set the distribution para as the last dim
        return x_hat, y_hat


class NBEATSStack(nn.Module):
    def __init__(
        self, block_num: int, layer_num: int, layer_width: int, expansion_coef_dim: int, input_chunk_len: int,
        target_len: int, g_type: GType, norm: str, dropout_rate: float, activation: str, **kwargs
    ):
        """PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.

        Parameters
        ----------
        block_num:   The number of blocks making up this stack.
        layer_num:   The number of fully connected layers preceding the final forking layers in each block.
        layer_width: The number of neurons that make up each fully connected layer in each block.
        expansion_coef_dim: The dimensionality of the waveform generator parameters, also known as expansion coefficients.
        input_chunk_len: The length of the input sequence fed to the model.
        target_len:      The length of the forecast of the model.
        g_type:          The function that is implemented by the waveform generators in each block.
        norm:            The norm to apply on the first block of this stack
        dropout_rate:    Dropout probability
        activation:      The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        stack_input of shape `(batch_size, input_chunk_len)`: Tensor containing the input sequence.

        Outputs
        -------
        stack_residual of shape `(batch_size, input_chunk_len)`: Tensor containing the 'backcast' of the block,
              which represents an approximation of `x` given the constraints of the functional space determined by `g`.
        stack_forecast of shape `(batch_size, output_chunk_len)`: Tensor containing the forward forecast of the stack.

        """
        super().__init__(**kwargs)

        self.input_chunk_len = input_chunk_len
        self.target_len = target_len
        self.dropout_rate = dropout_rate
        # self.norm = get_nn_attr(norm)
        self.activation = get_nn_attr(activation)

        if g_type == GType.GENERIC: # 水平
            self.blocks_list = [
                NBEATSBlock(
                    layer_num, layer_width, expansion_coef_dim, input_chunk_len, target_len, g_type,
                    norm=(norm if i==0 else ''), dropout_rate=self.dropout_rate, activation=self.activation,
                ) for i in range(block_num) # ^-- batch norm only on first block of first stack
            ]
        else: # same block instance is used for weight sharing
            interpretable_block = NBEATSBlock(
                layer_num, layer_width, expansion_coef_dim, input_chunk_len, target_len, g_type,
                norm=norm, dropout_rate=self.dropout_rate, activation=self.activation,
            )
            self.blocks_list = [interpretable_block] * block_num

        self.blocks = nn.ModuleList(self.blocks_list)

    def forward(self, x):
        # One forecast vector per parameter in the distribution
        stack_forecast = torch.zeros(
            x.shape[0], self.target_len, device=x.device, dtype=x.dtype,
        )

        residual = x # stack-level residual
        forecast = 0
        for block in self.blocks_list:
            block_backcast, block_forecast = block(residual)
            residual = residual - block_backcast
            forecast = forecast + block_forecast

        return residual, forecast


class NBEATSModel(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, generic_arch: bool, stack_num: int, block_num: int, layer_num: int,
        layer_widths: List[int], expansion_coef_dim: int, trend_poly_degree: int, input_chunk_len: int,
        output_chunk_len: int, target_len: int, norm: str, dropout_rate: float, activation: str, **kwargs
    ):
        """PyTorch module implementing the N-BEATS architecture.

        Parameters
        ----------
        output_dim: Number of output components in the target
        generic_arch: Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper
            (consisting of one trend and one waveform stack with appropriate waveform generator functions).
        stack_num: The number of stacks that make up the whole model. Only used if `generic_arch` is set to `True`.
        block_num: The number of blocks making up every stack.
        layer_num: The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths: Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `stack_num` and every entry in that list corresponds to
            the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coef_dim: The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_arch` is set to `True`.
        trend_poly_degree: The degree of the polynomial used as waveform generator in trend stacks.
            Only used if `generic_arch` is set to `False`.
        norm: the norm (batch usually) to apply on first block of the first stack
        dropout_rate: Dropout probability
        activation: The activation function of encoder/decoder intermediate layer.

        Inputs
        ------
        x: Tensor of shape `(batch_size, input_chunk_len)`, containing the input sequence.

        Outputs
        -------
        y: Tensor of shape `(batch_size, output_chunk_len, target_size/output_dim)`, containing the output of the NBEATS model.
        """

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_chunk_len = input_chunk_len
        self.output_chunk_len = output_chunk_len
        self.target_len = self.output_chunk_len * input_dim # ??
        self.input_chunk_len_multi = self.input_chunk_len * input_dim
        self.dropout_rate = dropout_rate
        # self.norm = get_nn_attr(norm)
        self.activation = get_nn_attr(activation)

        if generic_arch:
            self.stacks_list = [
                NBEATSStack(
                    block_num, layer_num, layer_widths[i], expansion_coef_dim,
                    self.input_chunk_len_multi, self.target_len, GType.GENERIC,
                    norm=(norm if i==0 else ''), dropout_rate=self.dropout_rate, activation=self.activation,
                ) for i in range(stack_num)  # batch norm only on first block of first stack
            ]
        else:
            trend_stack = NBEATSStack(
                block_num, layer_num, layer_widths[0], trend_poly_degree + 1, self.input_chunk_len_multi, self.target_len,
                GType.TREND, norm=norm, dropout_rate=self.dropout_rate, activation=self.activation,
            )
            waveform_stack = NBEATSStack(
                block_num, layer_num, layer_widths[1], -1, self.input_chunk_len_multi, self.target_len,
                GType.WAVEFORM, norm=norm, dropout_rate=self.dropout_rate, activation=self.activation,
            )
            self.stacks_list = [trend_stack, waveform_stack]

        self.stacks = nn.ModuleList(self.stacks_list)

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_head.requires_grad_(False)
        self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)

    def forward(self, x_in: Tuple):
        x, _ = x_in

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.input_chunk_len_multi, 1))
        x = x.squeeze(dim=2) # squeeze last dimension (because model is univariate)

        # One vector of length target_len per parameter in the distribution
        y = torch.zeros(x.shape[0], self.target_len, device=x.device, dtype=x.dtype,)

        residual = x # stack-level residual
        forecast = 0
        for stack in self.stacks_list:
            stack_backcast, stack_forecast = stack(residual)
            residual = residual - stack_backcast
            forecast = forecast + stack_forecast

        forecast = forecast.view(forecast.shape[0], self.output_chunk_len, self.input_dim)[:, :, : self.output_dim, :]
        return residual, forecast

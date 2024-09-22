'''
  完整版本的Tide模型
  2024-09-20

  Refs:
  1. https://github.com/ts-kim/RevIN/blob/master/RevIN.py
  2. https://github.com/google-research/google-research/blob/master/tide/models.py
  3. https://github.com/frinkleko/TiDE-Applications/blob/main/net.py
'''

import time
import torch
import numpy as np
from torch import nn, optim
from einops import rearrange
EPS = 1e-7


class RevIN(nn.Module):
    '''
    RevIN(Reversible Instance Normalization)解决分布偏移问题，提高模型的泛化能力。
    - 特征独立：每个特征有自己的仿射参数，允许对不同特征进行独立的缩放和偏移。
    - 参数效率：参数数量与特征数量成正比，而不是与序列长度或批量大小相关。
    - 可学习性：这些参数可以通过反向传播进行优化，使模型能够学习最佳的归一化参数。
    '''

    def __init__(self, num_features, eps=EPS):
        """
        num_features: the number of features or channels
        eps: a value added for numerical stability
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # 虽然代码片段中没有显式地展示学习过程，但这些参数是会被学习和修改的。
        # 1. 通过将这些张量定义为 nn.Parameter，它们自动成为模型的可学习参数。
        # 2. 所有nn.Parameter都会自动加入到模型参数列表，并在反向传播时算梯度。
        # 3. 当使用优化器(如Adam或SGD)对模型进行优化时，这些参数会被自动更新。

    def forward(self, x, mode):
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim-1)) # 下面统计到只剩下最后一个维度(...,x,y,z)->(...,1,1,z)
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach() # detach():无梯度张量, 不参与反向传播
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            # ^-- 先计算这种重要的统计量, 并保存(denorm的时候也用这)
            x = (x - self.mean) / (self.stdev + self.eps)
            x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
            x = x * self.stdev + self.mean
        else:
            raise NotImplementedError
        return x


class TideBlock(nn.Module):
    """
    TideBlock类实现了一个可自动设定隐藏层维度的TiDE块，包括两个线性层、残差连接、dropout和层归一化
    """
    def __init__(self, input_dim, output_dim, hidden_dim=0, dropout_rate=0.2, use_layernorm=True):
        super().__init__()

        def get_h_dim(input_dim, output_dim, hidden_dim):
            # 自动计算相关TideBlock的隐藏层维度，使用对数尺度
            if hidden_dim != 0:
                return hidden_dim
            else:
                log_in_dim = np.log2(input_dim)
                log_out_dim = np.log2(output_dim)
                return int(np.round(2 ** (log_in_dim/2 + log_out_dim/2)))

        the_hidden_dim = get_h_dim(input_dim, output_dim, hidden_dim)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, the_hidden_dim),
            nn.ReLU(),
            nn.Linear(the_hidden_dim, output_dim),
            nn.Dropout(dropout_rate),
        )
        self.skip = nn.Linear(input_dim, output_dim)  # 残差连接
        self.layer_norm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity

    def forward(self, x):
        "x: [batch_size, ts_len, input_dim (ie, ts_dim)] -> [batch_size, ts_len, output_dim]"
        out = self.dense(x) + self.skip(x)
        return self.layer_norm(out)


def make_coder(input_dim, output_dim, intmlp_dims, hidden_dims=[], dropout_rate=0.2, use_layernorm=True):
    """
    构建一个编码器或解码器，由多个 TideBlock 组成
    input_dim: 整个coder的输入维度(第一个TideBlock的入口)
    output_dim: 整个coder的输出维度(最后一个TideBlock的出口)
    intmlp_dims: 中间TideBlock的出入口维度列表
    hidden_dims: 各个TideBlock的隐藏状态维度列表
    """
    def get_bh_dim(hidden_dims=hidden_dims, *, i): # bh: block hidden
        # 计算相关TideBlock的隐藏层维度，使用对数尺度
        if not hasattr(hidden_dims, '__iter__'): # 是个标量(容错)
            hd = hidden_dims
        elif len(hidden_dims) > 0: # 非空列表
            hd = hidden_dims[i]
        else: # 是空列表
            hd = 0
        return hd

    all_dims = [input_dim] + intmlp_dims + [output_dim]
    layers = [TideBlock(all_dims[i-1], all_dims[i], get_bh_dim(hidden_dims, i=i), dropout_rate, use_layernorm)
              for i in range(1, len(all_dims))]
    return nn.Sequential(*layers)  # 使用 * 解包列表，创建 Sequential 模型


class ContactIn_Encoder(nn.Module):
    """
    这里实现了Dense Encoder, Feature Projection(projector), Residual part, 并对输入进行flatten和concat
    """
    def __init__(self, seq_len, pred_len, lookback_dim, covariate_dimr, attrib_dimr,
                 encoder_output_dim, encoder_intmlp_dims, encoder_hidden_dims=[],
                 projector_output_dim=0, projector_hidden_dim=0,
                 dropout_rate=0.2, revin=None, use_layernorm=True):
        self.seq_len = seq_len # L in the paper
        self.pred_len = pred_len # H in the paper

        self.encoder_input_dim = (self.seq_len + (self.seq_len + pred_len) * covariate_dimr + attrib_dimr) * lookback_dim
        self.encoder = make_coder(self.encoder_input_dim, encoder_output_dim,
                                  intmlp_dims=encoder_intmlp_dims, hidden_dims=encoder_hidden_dims,
                                  dropout_rate=dropout_rate, use_layernorm=use_layernorm)

        # use 1 block in feature projector accoding to the paper
        self.projector_input_dim = covariate_dimr * lookback_dim
        self.projector = TideBlock(self.projector_input_dim, projector_output_dim, projector_hidden_dim, dropout_rate, use_layernorm)
        self.residuar = nn.Linear(seq_len, pred_len) # TODO check

    def forward(self, lookback, covariate, attrib):
        '''
        lookback: [batch_size, seq_len, lookback_dim]
        covariate: [batch_size, seq_len+pred_len, lookback_dim*covariate_dimr]
        attrib: [batch_size, 1, lookback_dim*attrib_dimr]
        '''
        # project: [batch_size, seq_len+pred_len, lookback_dim*covariate_dimr] -> [batch_size, seq_len+pred_len, projector_output_dim]
        covariate_all = self.projector(covariate) # have the covariates projected (in a whole)(make it much smaller in dim)
        # covariate_past = covariate_all[:, :self.seq_len, :]  # [batch_size, seq_len, projector_output_dim]
        covariate_future = covariate_all[:, self.seq_len:, :]  # [batch_size, pred_len, projector_output_dim]
        # flat:将后两个维度合并:
        lookback_flat = rearrange(lookback, 'b l n -> b (l n)')  # [batch_size, seq_len, lookback_dim]
        covariate_flat = rearrange(covariate_all, 'b l r -> b (l r)') # [batch_size, seq_len+pred_len, covariate_dim]
        attrib_flat = rearrange(attrib, 'b l m -> b (l m)') # l==1, 相当于直接squeezed # [batch_size, 1, attrib_dim]
        # concat::
        concated = torch.cat([lookback_flat, covariate_flat, attrib_flat], dim=1)
        if revin is not None: concated = revin(concated)
        # encode::
        e = self.encoder(concated) # [batch_size, encoder_output_dim]
        # skip::
        lookback_tp = rearrange(lookback, 'b l n -> b n l')  # [batch_size, seq_len, lookback_dim] ->
        residual_tp = self.residuar(lookback) # [batch_size, lookback_dim, seq_len] -> [batch_size, lookback_dim, pred_len]
        residual = rearrange(residual_tp, 'b n h -> b h n')  # [batch_size, pred_len, lookback_dim]
        return e, covariate_future, residual
        # [batch_size, encoder_output_dim] [batch_size, pred_len, projector_output_dim] [batch_size, pred_len, lookback_dim]

class StackOut_Decoder(nn.Module):
    """
    这里实现了Dense Decoder, Temporal Decoder(tempor), 并做了unflatten和stack操作
    """
    def __init__(self, seq_len, pred_len, lookback_dim, encoder_output_dim, projector_output_dim,
                 decoder_output_dim, decoder_intmlp_dims, decoder_hidden_dims=[], tempor_hidden_dim=0,
                 dropout_rate=0.2, use_layernorm=True):
        self.seq_len = seq_len # L in the paper
        self.pred_len = pred_len # H in the paper

        #  [batch_size, pred_len, lookback_dim]
        self.decoder_input_dim = encoder_output_dim #  [batch_size, encoder_output_dim]
        self.decoder = make_coder(self.decoder_input_dim, decoder_output_dim,
                                  intmlp_dims=decoder_intmlp_dims, hidden_dims=decoder_hidden_dims,
                                  dropout_rate=dropout_rate, use_layernorm=use_layernorm)

        # use 1 block in tempor accoding to the paper
        self.tempor_input_dim = projector_output_dim + decoder_output_dim
        self.tempor = TideBlock(self.tempor_input_dim, lookback_dim, tempor_hidden_dim, dropout_rate, use_layernorm)

    def forward(self, e, covariate_future, residual):
        # decode::
        g = self.decoder(e) # [batch_size, encoder_output_dim] -> [batch_size, decoder_output_dim]
        # unflat::
        g = rearrange(g, 'b (h p) -> b h p', h=self.pred_len)
        # stack:[batch_size, pred_len, projector_output_dim+decoder_output_dim]:
        stacked = torch.cat([g, covariate_future], dim=-1) # concat on the last dim (item wise; ie, b h p+q)
        # decode again (temporal decoder):
        decoded = self.tempor(stacked) # -> [batch_size pred_len lookback_dim]
        # addout
        out = decoded + residual
        return out # [batch_size pred_len lookback_dim] TODO:-> [batch_size 1 lookback_dim]???


class TideModel(nn.Module):
    """
    完整版本的Tide模型
    """

    def __init__(self, seq_len, pred_len, lookback_dim, covariate_dimr, attrib_dimr,
                 encoder_output_dim, encoder_intmlp_dims, encoder_hidden_dims=[],
                 projector_output_dim=0, projector_hidden_dim=0,
                 decoder_output_dim=0, decoder_intmlp_dims=[], decoder_hidden_dims=[],
                 tempor_hidden_dim=0, dropout_rate=0.2, use_revin=True, use_layernorm=True):
        super().__init__()
        self.revin = RevIN(input_dim) if use_revin else nn.Identity()

        self.feedin = ContactIn_Encoder(seq_len, pred_len, lookback_dim, covariate_dimr, attrib_dimr,
                                        encoder_output_dim, encoder_intmlp_dims, encoder_hidden_dims,
                                        projector_output_dim, projector_hidden_dim, dropout_rate, revin, use_layernorm)
        self.stepout = StackOut_Decoder(seq_len, pred_len, lookback_dim, encoder_output_dim, projector_output_dim,
                                        decoder_output_dim, decoder_intmlp_dims, decoder_hidden_dims, tempor_hidden_dim,
                                        dropout_rate, use_layernorm)

    def forward(self, lookback, covariate, attrib):
        '`inputs` 包含过去数据、未来特征和时间序列索引'
        e, covariate_future, residual = ContactIn_Encoder(lookback, covariate, attrib)
        out = self.StackOut_Decoder(e, covariate_future, residual)
        out = self.revin(out, mode='denorm')
        return out

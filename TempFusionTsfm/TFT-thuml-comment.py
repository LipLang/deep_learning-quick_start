# https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py
# TFT (Temporal Fusion Transformer) 是一个用于多变量时间序列预测的深度学习模型

"""
Temporal Fusion Transformer (TFT)
================================

一个用于多变量时间序列预测的深度学习模型。主要创新点：
1. 变量处理机制：区分静态、观测和已知变量，分别处理
2. 时序处理：结合注意力机制和LSTM，捕获长短期依赖
3. 可解释性：通过变量选择网络和解释性注意力提供模型解释

Architecture Overview:
---------------------
1. 输入处理层：变量分类和特征嵌入
2. 特征选择层：静态编码和变量选择
3. 时序处理层：LSTM编码和自注意力机制
4. 输出层：前馈网络和残差连接

Implementation Details:
---------------------
- 变量分类：静态(static)、观测(observed)、已知(known)特征
- 特征选择：使用变量选择网络(VSN)动态选择重要特征
- 时序建模：结合LSTM和自注意力机制
- 解释性设计：提供特征重要性和时间注意力解释

Reference:
----------
- Paper: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- URL: https://arxiv.org/abs/1912.09363
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, TemporalEmbedding
from torch import Tensor
from typing import Optional
from collections import namedtuple

# 定义输入特征的类型
# static: time-independent features
#       : 静态特征，不随时间变化的特征（如用户性别、地理位置等）
# observed: time features of the past(e.g. predicted targets)
#         : 观测特征，随时间变化的历史特征（如历史销售量、温度等）
# known: known information about the past and future(i.e. time stamp)
#      : 已知特征，已知的未来信息（如时间戳、节假日等）
TypePos = namedtuple('TypePos', ['static', 'observed'])

# 数据集特征配置字典
# 为不同数据集定义其静态特征和观测特征的位置索引
# When you want to use new dataset, please add the index of 'static, observed' columns here.
# 'known' columns needn't be added, because 'known' inputs are automatically judged and provided by the program.
datatype_dict = {
    'ETTh1': TypePos([], [x for x in range(7)]),  # ETTh1数据集没有静态特征，有7个观测特征
    'ETTm1': TypePos([], [x for x in range(7)])   # ETTm1数据集配置相同
}

def get_known_len(embed_type, freq):
    """获取已知特征（时间特征）的维度

    Args:
        embed_type: 嵌入类型，'fixed'或'timeF'
        freq: 时间频率，如'h'(小时),'d'(天)等

    Returns:
        int: 时间特征的维度
    """
    if embed_type != 'timeF':
        if freq == 't':
            return 5  # 分钟级数据包含5个时间特征
        else:
            return 4  # 其他频率包含4个时间特征
    else:
        # timeF嵌入方式下不同时间频率的特征维度
        freq_map = {'h': 4, 't': 5, 's': 6,  # 小时、分钟、秒
                   'm': 1, 'a': 1, 'w': 2,   # 月、年、周
                   'd': 3, 'b': 3}           # 日、工作日
        return freq_map[freq]

class TFTTemporalEmbedding(TemporalEmbedding):
    """时间特征嵌入模块

    将时间信息（年、月、日、小时等）转换为向量表示
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TFTTemporalEmbedding, self).__init__(d_model, embed_type, freq)

    def forward(self, x):
        """时间特征编码转换"""
        x = x.long()
        # 分别对不同时间尺度特征进行嵌入
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # 组合所有时间特征的嵌入
        embedding_x = torch.stack(
            [month_x, day_x, weekday_x, hour_x, minute_x], dim=-2
        ) if hasattr(self, 'minute_embed') else torch.stack(
            [month_x, day_x, weekday_x, hour_x], dim=-2
        )
        return embedding_x

class TFTTimeFeatureEmbedding(nn.Module):
    """时间特征的线性嵌入模块

    使用线性变换将时间特征映射到高维空间
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TFTTimeFeatureEmbedding, self).__init__()
        d_inp = get_known_len(embed_type, freq)
        # 为每个时间特征创建一个线性变换层
        self.embed = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(d_inp)])

    def forward(self, x):
        # 对每个时间特征进行线性变换并堆叠
        return torch.stack([embed(x[:,:,i].unsqueeze(-1)) for i, embed in enumerate(self.embed)], dim=-2)

class TFTEmbedding(nn.Module):
    """TFT模型的特征嵌入层

    处理三种类型的输入特征：
    1. 静态特征
    2. 观测特征
    3. 已知特征（时间特征）
    """
    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.pred_len = configs.pred_len
        # 获取特征位置信息
        self.static_pos = datatype_dict[configs.data].static
        self.observed_pos = datatype_dict[configs.data].observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        # 创建各类特征的嵌入层
        self.static_embedding = nn.ModuleList(
            [DataEmbedding(1, configs.d_model, dropout=configs.dropout)
             for _ in range(self.static_len)]
        ) if self.static_len else None

        self.observed_embedding = nn.ModuleList(
            [DataEmbedding(1, configs.d_model, dropout=configs.dropout)
             for _ in range(self.observed_len)]
        )

        # 时间特征嵌入层
        self.known_embedding = (
            TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq)
            if configs.embed != 'timeF' else
            TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Args:
            x_enc: 编码器输入序列
            x_mark_enc: 编码器时间特征
            x_dec: 解码器输入序列
            x_mark_dec: 解码器时间特征

        Returns:
            static_input: 静态特征嵌入
            observed_input: 观测特征嵌入
            known_input: 时间特征嵌入
        """
        # 处理静态特征
        if self.static_len:
            # static_input: [B,C,d_model]
            static_input = torch.stack([
                embed(x_enc[:,:1,self.static_pos[i]].unsqueeze(-1), None).squeeze(1)
                for i, embed in enumerate(self.static_embedding)
            ], dim=-2)
        else:
            static_input = None

        # 处理观测特征 observed_input: [B,T,C,d_model]
        observed_input = torch.stack([
            embed(x_enc[:,:,self.observed_pos[i]].unsqueeze(-1), None)
            for i, embed in enumerate(self.observed_embedding)
        ], dim=-2)

        # 处理时间特征
        x_mark = torch.cat([x_mark_enc, x_mark_dec[:,-self.pred_len:,:]], dim=-2)
        # known_input: [B,T,C,d_model]
        known_input = self.known_embedding(x_mark)

        return static_input, observed_input, known_input

class GLU(nn.Module):
    """门控线性单元

    通过门控机制控制信息流动，增强模型的非线性表达能力
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GateAddNorm(nn.Module):
    """门控加法和归一化层

    执行以下操作：
    1. 门控处理输入
    2. 添加跳跃连接
    3. 层归一化
    """
    def __init__(self, input_size, output_size):
        super(GateAddNorm, self).__init__()
        self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        """前向传播过程，包括门控、残差连接和归一化"""
        x = self.glu(x)  # 门控处理
        x = x + skip_a   # 残差连接
        return self.layer_norm(self.projection(x))  # 投影和归一化

class GRN(nn.Module):
    """门控残差网络(Gated Residual Network)

    一个功能强大的特征处理模块，包含：
    1. 多层特征转换
    2. 上下文信息整合
    3. 门控机制
    4. 残差连接
    """
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        """
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            hidden_size: 隐藏层维度（默认等于input_size）
            context_size: 上下文特征维度（可选）
            dropout: dropout率
        """
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)         # 输入转换层
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None  # 上下文处理层
        self.lin_i = nn.Linear(hidden_size, hidden_size)        # 中间转换层
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)       # 门控和归一化

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        """
        Args:
            a: 主要输入特征 [batch_size, seq_len, input_size]
            c: 上下文特征 [batch_size, context_size]（可选）
        # a: [B,T,d], c: [B,d]
        """
        x = self.lin_a(a)  # 特征转换
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)  # 融合上下文信息
        x = F.elu(x)       # 非线性激活
        x = self.lin_i(x)  # 进一步转换
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))  # 门控处理和残差连接

class VariableSelectionNetwork(nn.Module):
    """变量选择网络

    动态选择重要变量的关键模块，提供可解释性：
    1. 计算变量重要性权重
    2. 独立处理每个变量
    3. 加权组合得到最终表示
    """
    def __init__(self, d_model, variable_num, dropout=0.0):
        super(VariableSelectionNetwork, self).__init__()
        # 处理所有变量的联合特征
        self.joint_grn = GRN(
            d_model * variable_num,  # 所有变量特征拼接
            variable_num,            # 输出变量数量的权重
            hidden_size=d_model,
            context_size=d_model,
            dropout=dropout
        )
        # 独立处理每个变量的GRN
        self.variable_grns = nn.ModuleList([
            GRN(d_model, d_model, dropout=dropout)
            for _ in range(variable_num)
        ])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        """
        Args:
            x: 输入特征 [batch_size, seq_len/1, num_vars, d_model]
            context: 上下文特征 [batch_size, d_model]

        Returns:
            selection_result: 选择后的特征表示
        """
        # x: [B,T,C,d] or [B,C,d]
        # selection_weights: [B,T,C] or [B,C]
        # x_processed: [B,T,d,C] or [B,d,C]
        # selection_result: [B,T,d] or [B,d]

        # 计算变量重要性权重
        x_flattened = torch.flatten(x, start_dim=-2)
        selection_weights = self.joint_grn(x_flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)

        # 独立处理每个变量并加权组合
        x_processed = torch.stack([
            grn(x[...,i,:]) for i, grn in enumerate(self.variable_grns)
        ], dim=-1)

        # 加权求和得到最终表示
        selection_result = torch.matmul(
            x_processed, selection_weights.unsqueeze(-1)
        ).squeeze(-1)

        return selection_result

class StaticCovariateEncoder(nn.Module):
    """静态协变量编码器

    处理静态特征，生成多个上下文向量用于后续处理：
    1. 变量选择上下文
    2. 时序编码上下文
    3. 增强层上下文
    """
    def __init__(self, d_model, static_len, dropout=0.0):
        super(StaticCovariateEncoder, self).__init__()
        # 静态特征选择网络
        self.static_vsn = VariableSelectionNetwork(
            d_model, static_len
        ) if static_len else None
        # 生成四个上下文向量的GRN
        self.grns = nn.ModuleList([
            GRN(d_model, d_model, dropout=dropout)
            for _ in range(4)
        ])

    def forward(self, static_input):
        """
        Args:
            static_input: 静态特征输入 [batch_size, num_static, d_model]

        Returns:
            list: 四个上下文向量，用于不同的处理阶段
        """
        # static_input: [B,C,d]
        if static_input is not None:
            static_features = self.static_vsn(static_input)
            return [grn(static_features) for grn in self.grns]
        else:
            return [None] * 4

class InterpretableMultiHeadAttention(nn.Module):
    """可解释的多头注意力机制

    相比标准Transformer的注意力机制的改进：
    1. 单值注意力：每个时间步只输出一个值
    2. 因果注意力：只关注历史信息
    3. 可解释性：提供时间注意力权重
    """
    def __init__(self, configs):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_heads = configs.n_heads
        assert configs.d_model % configs.n_heads == 0
        self.d_head = configs.d_model // configs.n_heads

        # Q,K,V转换层
        self.qkv_linears = nn.Linear(
            configs.d_model,
            (2 * self.n_heads + 1) * self.d_head,  # Q,K各n_heads个头，V只有1个
            bias=False
        )

        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5

        # 创建因果注意力掩码
        example_len = configs.seq_len + configs.pred_len
        self.register_buffer(
            "mask",
            torch.triu(
                torch.full((example_len, example_len), float('-inf')), 1
            )
        )

    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, d_model]

        Returns:
            out: 注意力处理后的序列
        """
        # Q,K,V are all from x
        B, T, d_model = x.shape

        # 生成Q,K,V
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split(
            (self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head),
            dim=-1
        )

        # 重塑Q,K,V的形状
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.d_head)

        # 计算注意力分数
        attention_score = torch.matmul(
            q.permute(0, 2, 1, 3),  # [B, n_heads, T, d_head]
            k.permute(0, 2, 3, 1)   # [B, n_heads, d_head, T]
        )  # [B,n,T,T]

        attention_score.mul_(self.scale)
        attention_score = attention_score + self.mask  # 添加因果掩码
        attention_prob = F.softmax(attention_score, dim=3)  # [B,n,T,T]

        # 注意力加权和平均
        attention_out = torch.matmul(attention_prob, v.unsqueeze(1))  # [B,n,T,d]
        attention_out = torch.mean(attention_out, dim=1)  # [B,T,d]

        # 输出投影
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)  # [B,T,d]
        return out

class TemporalFusionDecoder(nn.Module):
    """时序融合解码器

    TFT模型的核心解码器，整合各种信息：
    1. 历史信息编码
    2. 未来信息编码
    3. 静态信息增强
    4. 时序自注意力
    """
    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.pred_len = configs.pred_len

        # 双向LSTM编码器
        self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.future_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)

        # 各种处理层
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model)
        self.enrichment_grn = GRN(
            configs.d_model, configs.d_model,
            context_size=configs.d_model,
            dropout=configs.dropout
        )
        self.attention = InterpretableMultiHeadAttention(configs)
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model)
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, history_input, future_input, c_c, c_h, c_e):
        """
        Args:
            history_input: 历史输入
            future_input: 未来已知输入
            c_c, c_h, c_e: 静态特征生成的上下文向量
        """
        # history_input, future_input: [B,T,d]
        # c_c, c_h, c_e: [B,d]
        # LSTM编码
        c = (c_c.unsqueeze(0), c_h.unsqueeze(0)) if c_c is not None and c_h is not None else None
        historical_features, state = self.history_encoder(history_input, c)
        future_features, _ = self.future_encoder(future_input, state)

        # Skip connection       # 特征融合
        temporal_input = torch.cat([history_input, future_input], dim=1)
        temporal_features = torch.cat([historical_features, future_features], dim=1)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input) # [B,T,d]

        # Static enrichment 静态特征增强
        enriched_features = self.enrichment_grn(temporal_features, c_e)  # [B,T,d]

        # 时序自注意力
        # Temporal self-attention
        attention_out = self.attention(enriched_features)  # [B,T,d]
        # Don't compute historical loss
        attention_out = self.gate_after_attention(
            attention_out[:,-self.pred_len:],
            enriched_features[:,-self.pred_len:]
        )

        # 位置式前馈网络# Position-wise feed-forward
        out = self.position_wise_grn(attention_out)  # [B,T,d]

        # 最终残差连接        # Final skip connection
        out = self.gate_final(out, temporal_features[:,-self.pred_len:])
        return self.out_projection(out)

class Model(nn.Module):
    """TFT完整模型

    整合所有组件的端到端模型，包括：
    1. 特征嵌入
    2. 变量选择
    3. 时序处理
    4. 预测输出
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # 获取特征数量
        self.static_len = len(datatype_dict[configs.data].static)
        self.observed_len = len(datatype_dict[configs.data].observed)
        self.known_len = get_known_len(configs.embed, configs.freq)

        # 主要组件
        self.embedding = TFTEmbedding(configs)
        self.static_encoder = StaticCovariateEncoder(configs.d_model, self.static_len)
        self.history_vsn = VariableSelectionNetwork(
            configs.d_model,
            self.observed_len + self.known_len
        )
        self.future_vsn = VariableSelectionNetwork(
            configs.d_model,
            self.known_len
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(configs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """预测函数

        完整的前向传播过程：
        1. 数据归一化
        2. 特征嵌入
        3. 变量选择
        4. 时序融合
        5. 生成预测
        """
        # Normalization from Non-stationary Transformer
        # 数据归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # 特征嵌入
        # static_input: [B,C,d], observed_input:[B,T,C,d], known_input: [B,T,C,d]
        static_input, observed_input, known_input = self.embedding(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )

        # 静态特征编码
        # c_s,...,c_e: [B,d]
        c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        # 变量选择
        history_input = torch.cat(
            [observed_input, known_input[:,:self.seq_len]], dim=-2
        )
        future_input = known_input[:,self.seq_len:]
        history_input = self.history_vsn(history_input, c_s)
        future_input = self.future_vsn(future_input, c_s)

        # TFT main procedure after variable selection
        # 时序融合和预测
        # history_input: [B,T,d], future_input: [B,T,d]
        dec_out = self.temporal_fusion_decoder(
            history_input, future_input, c_c, c_h, c_e
        )

        # 反归一化
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """模型前向传播

        根据任务类型选择相应的处理流程
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec) # [B,pred_len,C]
            dec_out = torch.cat([torch.zeros_like(x_enc), dec_out], dim=1)
            return dec_out #  [B,T,d]
        return None

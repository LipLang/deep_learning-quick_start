In additional to `TiDE` and `nHits`,
[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
seems also promising. 

`Just Do It!`


1. TFT 模型的思想：

TFT 是一种专门为多变量时间序列预测设计的深度学习模型。它的主要思想是结合了几种不同的机制来处理复杂的时间序列预测任务：

- 处理不同类型的输入（静态特征、已知的动态特征、未知的动态特征）
- 使用注意力机制来捕捉长期依赖关系
- 通过变量选择网络来识别重要的驱动因素
- 使用门控线性单元 (GLU) 来控制信息流
- 采用多头注意力机制来捕捉不同时间尺度的模式

2. 网络架构：

TFT 的网络架构包括以下主要组件：

a) 输入处理层：
   - 静态协变量编码器
   - 时变已知输入编码器
   - 时变未知输入编码器

b) 变量选择网络：
   - 用于识别重要特征

c) 静态协变量编码器：
   - 处理静态特征

d) 时间处理：
   - 位置编码

e) 长期和短期处理层：
   - 基于 LSTM 的序列处理
   - 门控残差连接

f) 时间注意力层：
   - 多头注意力机制

g) 位置前馈层

h) 量化输出层

------

以下是一个简化版的 TFT 模型实现：

```python
import torch
import torch.nn as nn

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.flattened_grn = GatedResidualNetwork(input_size, hidden_size)
        self.variable_grns = nn.ModuleList([GatedResidualNetwork(hidden_size, hidden_size) for _ in range(input_size)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        flat = torch.flatten(inputs, start_dim=-2)
        weights = self.flattened_grn(flat)
        weights = weights.view([-1, self.input_size, self.hidden_size])
        weights = self.softmax(weights)
        transformed_inputs = torch.stack([grn(inputs[..., i]) for i, grn in enumerate(self.variable_grns)], dim=-1)
        return torch.sum(transformed_inputs * weights, dim=-1)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, self.output_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_size + hidden_size, self.output_size)
        self.skip_layer = nn.Linear(input_size, self.output_size)

    def forward(self, x):
        hidden = self.layer1(x)
        hidden = self.elu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        gating_input = torch.cat([x, hidden], dim=-1)
        gate = torch.sigmoid(self.gate(gating_input))
        output = gate * self.layer3(hidden) + (1 - gate) * self.skip_layer(x)
        return output

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_static, num_dynamic_real, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.static_vsn = VariableSelectionNetwork(num_static, hidden_size)
        self.dynamic_vsn = VariableSelectionNetwork(num_dynamic_real, hidden_size)
        self.static_grn = GatedResidualNetwork(hidden_size, hidden_size)
        self.dynamic_grn = GatedResidualNetwork(hidden_size, hidden_size)
        self.position_encoding = nn.Embedding(100, hidden_size)  # Assuming max sequence length of 100
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, static_inputs, dynamic_inputs):
        batch_size, seq_len, _ = dynamic_inputs.shape
        
        # Process static inputs
        static_embeddings = self.static_vsn(static_inputs)
        static_embeddings = self.static_grn(static_embeddings)
        static_embeddings = static_embeddings.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Process dynamic inputs
        dynamic_embeddings = self.dynamic_vsn(dynamic_inputs)
        dynamic_embeddings = self.dynamic_grn(dynamic_embeddings)
        
        # Combine static and dynamic embeddings
        embeddings = static_embeddings + dynamic_embeddings
        
        # Add positional encoding
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(embeddings.device)
        embeddings = embeddings + self.position_encoding(positions)
        
        # LSTM layer
        lstm_output, _ = self.lstm(embeddings)
        
        # Self-attention layer
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Final layer
        output = self.final_layer(attn_output)
        
        return output.squeeze(-1)

# Example usage
num_static = 5
num_dynamic_real = 10
hidden_size = 64
num_heads = 4

model = TemporalFusionTransformer(num_static, num_dynamic_real, hidden_size, num_heads)

# Generate some dummy data
batch_size = 32
seq_len = 50
static_inputs = torch.randn(batch_size, num_static)
dynamic_inputs = torch.randn(batch_size, seq_len, num_dynamic_real)

# Forward pass
output = model(static_inputs, dynamic_inputs)
print(output.shape)  # Should be [batch_size, seq_len]
```

这个实现包含了 TFT 的主要组件，包括变量选择网络、门控残差网络、LSTM 层和多头注意力机制。
但请注意，这是一个简化版本，完整的 TFT 模型还包括其他一些细节，如量化器、元学习器等。
要使用这个模型，您需要准备好静态输入和动态输入，然后将它们传递给模型。
模型将输出一个形状为 `[batch_size, seq_len]` 的张量，表示每个时间步的预测。

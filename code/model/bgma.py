# Bidirectional Gated Mamba Attention for Video Object Segmentation
# BGMA: Bidirectional Gated Mamba Attention

from torch import nn
import torch
import torch.nn.functional as F
from mamba_ssm import Mamba  # <--- 导入Mamba


# ============================================================================
# 原始的 ChannelAttention 实现 (注释保留，方便日后找回)
# ============================================================================
# class ChannelAttention(nn.Module):
#     def __init__(self, input_channels, internal_neurons):
#         super(ChannelAttention, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
#         self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
#         self.input_channels = input_channels
# 
#     def forward(self, inputs):
#         x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
#         x1 = self.fc1(x1)
#         x1 = F.relu(x1, inplace=True)
#         x1 = self.fc2(x1)
#         x1 = torch.sigmoid(x1)
#         
#         x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
#         x2 = self.fc1(x2)
#         x2 = F.relu(x2, inplace=True)
#         x2 = self.fc2(x2)
#         x2 = torch.sigmoid(x2)
#         
#         x = x1 + x2
#         x = x.view(-1, self.input_channels, 1, 1)
#         return x


# ============================================================================
# 新的基于 Mamba 的通道注意力模块
# ============================================================================
class MambaChannelAttention(nn.Module):
    def __init__(self, input_channels, d_model=32, d_state=16):
        """
        基于 Mamba (S6) 的通道注意力模块.
        
        Args:
            input_channels: 输入通道数 (C)
            d_model: Mamba 内部模型维度 (建议较小值如32/64)
            d_state: Mamba 状态维度 N (控制状态空间大小)
        """
        super().__init__()
        self.input_channels = input_channels
        
        # 将通道维当作序列长度 L=C，每个位置特征维度为1，投影到d_model
        # -- 投影解耦：为 avg 和 max 分支设置独立的输入投影层 --
        self.input_proj_avg = nn.Linear(1, d_model)
        self.input_proj_max = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)
        
        # 双向 Mamba：前向 + 后向
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展因子
        )
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

        # 门控融合头（后融合）：基于 avg/max 通道统计预测 α，初始化为 0.5
        self.gate = nn.Linear(2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W)
        batch_size = inputs.size(0)

        # --------------------------------------------------------------------
        # 原双分支版本：avg 分支 + max 分支，各自过 Bi-Mamba 后再相加
        # --------------------------------------------------------------------
        # 1. 全局池化：平均池化 + 最大池化
        x_avg = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))  # (B, C, 1, 1)
        x_max = F.adaptive_max_pool2d(inputs, output_size=(1, 1))  # (B, C, 1, 1)

        # 2. 形状调整：(B, C, 1, 1) -> (B, C, 1)
        x_avg = x_avg.squeeze(-1)  # (B, C, 1)
        x_max = x_max.squeeze(-1)  # (B, C, 1)

        # 2.1 基于通道统计的门控 α（每样本、每通道），初始化时为 0.5
        gate_in = torch.cat([x_avg, x_max], dim=-1)   # (B, C, 2)
        alpha = torch.sigmoid(self.gate(gate_in))     # (B, C, 1)

        # 3. 投影到 Mamba 的模型维度（使用解耦的投影层）
        x_avg = self.input_proj_avg(x_avg)  # (B, C, d_model)
        x_max = self.input_proj_max(x_max)  # (B, C, d_model)

        # 4. 双向 Mamba 处理
        # 前向
        y_avg_fwd = self.mamba_forward(x_avg)
        y_max_fwd = self.mamba_forward(x_max)
        # 反向（翻转序列维度）
        y_avg_bwd = self.mamba_backward(torch.flip(x_avg, dims=[1]))
        y_max_bwd = self.mamba_backward(torch.flip(x_max, dims=[1]))
        # 翻回对齐
        y_avg_bwd = torch.flip(y_avg_bwd, dims=[1])
        y_max_bwd = torch.flip(y_max_bwd, dims=[1])

        # 5. 融合前向+后向（各分支内部）
        y_avg = y_avg_fwd + y_avg_bwd
        y_max = y_max_fwd + y_max_bwd

        # 6. 门控后融合：y = α·y_avg + (1-α)·y_max
        #    alpha 形状 (B,C,1) —— 依最后一维广播到 d_model
        y = alpha * y_avg + (1.0 - alpha) * y_max  # (B, C, d_model)
        
        # 7. 投影回标量并应用 sigmoid
        attention_weights = self.output_proj(y)  # (B, C, 1)
        attention_weights = torch.sigmoid(attention_weights)
        
        # 8. 调整形状为 (B, C, 1, 1) 以便与输入特征相乘
        attention_weights = attention_weights.unsqueeze(-1)  # (B, C, 1, 1)
        
        return attention_weights


class BGMABlock(nn.Module):

    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        
        # 自适应选择 Mamba 参数
        if in_channels >= 512:
            d_model, d_state = 64, 32
        elif in_channels >= 256:
            d_model, d_state = 32, 16
        else:
            d_model, d_state = 16, 8
            
        # 使用自适应参数的 Mamba 通道注意力模块
        self.ca = MambaChannelAttention(input_channels=in_channels, d_model=d_model, d_state=d_state)
        
        # 如果想要切换回原来的实现，取消下面这行的注释，并注释上面的 MambaChannelAttention
        # self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # Save original input for residual connection
        identity = inputs
        
        # Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        
        # Add residual connection to prevent feature collapse
        out = out + identity
        
        return out



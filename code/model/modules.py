"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask.
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
from model.bgma import BGMABlock


class MSFBlock(nn.Module):
    """Multi-Scale Selective Fusion Block adapted for 3 inputs and grouped tensors"""
    def __init__(self, in_channels):
        super(MSFBlock, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            GConv2D(out_channels, out_channels, 1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=3)
        self.sigmoid = nn.Sigmoid()

        # SE modules for 3 scales (g16, g8, g4)
        self.SE1 = GConv2D(in_channels, in_channels, 1)
        self.SE2 = GConv2D(in_channels, in_channels, 1)
        self.SE3 = GConv2D(in_channels, in_channels, 1)

    def forward(self, x0, x1, x2):
        # x0/x1/x2: (B, num_objects, C, H, W)
        batch_size, num_objects = x0.shape[:2]

        # Global Average Pooling + 1x1 conv for each scale
        y0_weight = self.SE1(self.gap(x0))  # (B, num_objects, C, 1, 1)
        y1_weight = self.SE2(self.gap(x1))  # (B, num_objects, C, 1, 1)
        y2_weight = self.SE3(self.gap(x2))  # (B, num_objects, C, 1, 1)

        # Concatenate along scale dimension: (B, num_objects, C, 3, 1)
        weight = torch.cat([y0_weight, y1_weight, y2_weight], 3)

        # Sigmoid + Softmax for channel-wise scale competition
        weight = self.softmax(self.sigmoid(weight))

        # Extract weights for each scale
        y0_weight = weight[:, :, :, 0:1, :]  # (B, num_objects, C, 1, 1)
        y1_weight = weight[:, :, :, 1:2, :]  # (B, num_objects, C, 1, 1)
        y2_weight = weight[:, :, :, 2:3, :]  # (B, num_objects, C, 1, 1)

        # Weighted fusion
        x_att = y0_weight * x0 + y1_weight * x1 + y2_weight * x2

        return self.project(x_att)


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + MU-GRU (Motion-Unaware GRU)
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        # MSF block for multi-scale selective fusion
        self.msf_fusion = MSFBlock(mid_dim)

        # Motion gate convolution: converts motion delta to update_gate logits
        self.motion_gate_conv = GConv2D(mid_dim, hidden_dim, kernel_size=3, padding=1)

        # Standard 3x3 for gate generator
        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)
        
        # Initialize motion gate conv with smaller weights to start conservatively
        nn.init.xavier_normal_(self.motion_gate_conv.weight, gain=0.1)

        # Temporal buffer for previous fused representation
        self.prev_fused = None

    def reset_temporal_buffer(self):
        """Reset cached fused features between video sequences."""
        self.prev_fused = None

    def forward(self, g, h):
        # Multi-scale selective fusion: g16 + g8 downsampled + g4 downsampled through MSF
        g16_feat = self.g16_conv(g[0])
        g8_feat = self.g8_conv(downsample_groups(g[1], ratio=1/2))
        g4_feat = self.g4_conv(downsample_groups(g[2], ratio=1/4))

        # Use MSF block for selective fusion instead of simple addition
        fused = self.msf_fusion(g16_feat, g8_feat, g4_feat)

        # Motion-Unaware GRU: Compute motion delta for update_gate modulation  
        if (self.prev_fused is None) or (not torch.any(h)):
            # First frame or sequence start: no motion modulation
            motion_logits = torch.zeros_like(fused[:,:,:self.hidden_dim])
        else:
            # Calculate motion delta (frame difference)
            g_delta = fused - self.prev_fused
            # End-to-end learnable motion gate: g_delta -> motion_logits
            motion_logits = self.motion_gate_conv(g_delta)

        # Cache current fused feature for next frame (detach to stop gradient-through-time)
        self.prev_fused = fused.detach()

        # Standard GRU computation with motion-modulated update gate
        g = torch.cat([fused, h], 2)

        # Generate all gate logits
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        
        # Motion-Unaware Update Gate: enhanced by motion logits
        update_gate_logits = values[:,:,self.hidden_dim:self.hidden_dim*2]
        update_gate = torch.sigmoid(update_gate_logits + motion_logits)
        
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])

        # Final GRU update with motion-aware gating
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()

        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g)

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 原始 XMem KeyEncoder（标准 ResNet50，无可形变/空洞改动）
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)

        # Add BGMA module after upsampling (with residual connection fixed)
        self.bgma = BGMABlock(in_channels=g_up_dim, out_channels=g_up_dim, channelAttention_reduce=4)
        # Add CBAM module after upsampling
        # self.cbam = CBAM(g_up_dim)

        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)

        # Upsample
        g = upsample_groups(up_g, ratio=self.scale_factor)

        # Apply BGMA attention after upsampling
        # BGMA doesn't support grouped tensors, so flatten and view
        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)
        g = self.bgma(g)
        g = g.view(batch_size, num_objects, *g.shape[1:])

        # Fusion with skip connection
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None

        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None

        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

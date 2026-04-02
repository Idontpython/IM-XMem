"""
resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

try:
    from torchvision.ops import DeformConv2d as TVDeformConv2d
except Exception:
    TVDeformConv2d = None

def load_weights_add_extra_dim(target, source_state, extra_dim=1):
    new_dict = OrderedDict()

    for k_target, v_target in target.state_dict().items():
        if 'num_batches_tracked' in k_target:
            continue

        k_source = k_target
        # 适配可形变卷积的权重key
        if k_target.endswith('.conv.weight'):
            k_source = k_target.replace('.conv.weight', '.weight')
        elif k_target.endswith('.conv.bias'):
            k_source = k_target.replace('.conv.bias', '.bias')

        if k_source in source_state:
            v_source = source_state[k_source]

            if v_target.shape != v_source.shape:
                # 处理第一个卷积层输入通道不匹配的问题
                if 'conv1.weight' in k_target and extra_dim > 0 and v_target.shape[1] == v_source.shape[1] + extra_dim:
                    c, _, w, h = v_target.shape
                    pads = torch.zeros((c, extra_dim, w, h), device=v_source.device)
                    nn.init.orthogonal_(pads)
                    v_source = torch.cat([v_source, pads], 1)
                else:
                    # 其他形状不匹配的层则跳过
                    print(f'Skipping {k_target} due to shape mismatch: src {v_source.shape}, dst {v_target.shape}')
                    continue
            
            new_dict[k_target] = v_source
    
    # 使用 strict=False 加载权重, 忽略缺失的key (比如可形变卷积的offset层)
    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class DeformableConv2d(nn.Module):
    """简易可形变 3x3 卷积封装，内部带 offset 分支。
    若 torchvision.ops 不可用，则回退为普通卷积，保证可运行。
    """
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, groups=1,
                 deformable_groups=1, bias=False):
        super().__init__()
        self.use_deform = TVDeformConv2d is not None
        if self.use_deform:
            k = 3
            off_ch = deformable_groups * 2 * k * k
            self.offset = nn.Conv2d(in_ch, off_ch, kernel_size=k, stride=stride,
                                    padding=padding, dilation=dilation, bias=True)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            # 移除 deformable_groups 以兼容旧版 torchvision
            self.conv = TVDeformConv2d(in_ch, out_ch, kernel_size=k, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.offset = None
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.offset is not None:
            off = self.offset(x)
            return self.conv(x, off)
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 use_deformable=False, deformable_groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # BasicBlock 保持使用普通卷积，不使用可形变卷积（ValueEncoder 不需要修改）
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 use_deformable=False, deformable_groups=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if use_deformable:
            self.conv2 = DeformableConv2d(planes, planes, stride=stride, padding=dilation,
                                          dilation=dilation, deformable_groups=deformable_groups, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                                   padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0,
                 use_deform_layer1=False, layer3_dilation=1, deformable_groups=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3+extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.use_deform_layer1 = use_deform_layer1
        self.layer3_dilation = layer3_dilation
        self.deformable_groups = deformable_groups

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=1, layer_index=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, layer_index=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=self.layer3_dilation, layer_index=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=1, layer_index=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, layer_index=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        use_deform = (layer_index == 1 and self.use_deform_layer1)
        layers = [block(self.inplanes, planes, stride, downsample, dilation=dilation,
                        use_deformable=use_deform, deformable_groups=self.deformable_groups)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                use_deformable=use_deform, deformable_groups=self.deformable_groups))

        return nn.Sequential(*layers)

def resnet18(pretrained=True, extra_dim=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_dim)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet18']), extra_dim)
    return model

def resnet50(pretrained=True, extra_dim=0, use_deform_layer1=False, layer3_dilation=1, deformable_groups=1):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_dim,
                   use_deform_layer1=use_deform_layer1, layer3_dilation=layer3_dilation,
                   deformable_groups=deformable_groups)
    if pretrained:
        load_weights_add_extra_dim(model, model_zoo.load_url(model_urls['resnet50']), extra_dim)
    return model


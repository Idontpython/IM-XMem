import os
import random
import numpy as np
from PIL import Image

import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# 添加路径以导入XMem的工具函数
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))
if _code_dir not in sys.path:
    sys.path.append(_code_dir)
from util.tensor_util import pad_divide_by
from dataset.range_transform import im_normalization

class S2MDataset(Dataset):
    def __init__(self, root_path, data_file, size=480):
        """
        初始化数据集
        :param root_path: 项目根目录 (e.g., /home/xushutan/IMXmem)
        :param data_file: 数据清单文件的路径 (e.g., datasets/train.txt)
        :param size: 图像预处理后的尺寸
        """
        self.root_path = root_path
        self.size = size
        with open(os.path.join(root_path, data_file), 'r') as f:
            self.samples = f.readlines()

        # 定义图像和掩码的预处理 - 使用与推理时完全一致的方式
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size), Image.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 从清单中解析图片和掩码的相对路径
        img_path, mask_path = self.samples[idx].strip().split()
        
        # 构建绝对路径
        full_img_path = os.path.join(self.root_path, img_path)
        full_mask_path = os.path.join(self.root_path, mask_path)

        # 读取图片和掩码
        image = Image.open(full_img_path).convert('RGB')
        mask = Image.open(full_mask_path).convert('L') # 转为灰度图

        # 获取图像尺寸
        width, height = image.size
        
        # 预处理掩码并转换为numpy
        mask = np.array(mask)
        # 将掩码二值化 (前景为1, 背景为0)
        mask = (mask > 0).astype(np.uint8)

        h, w = mask.shape

        # --- 模拟交互：增强正负涂鸦 ---
        pos_scribble_map = np.zeros_like(mask, dtype=np.uint8)
        neg_scribble_map = np.zeros_like(mask, dtype=np.uint8)

        positive_points_coords = np.argwhere(mask > 0)
        if len(positive_points_coords) > 0:
            # 计算边界与扩展区域
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_large = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask, kernel_small, iterations=1)
            boundary_map = (mask - eroded).clip(min=0)
            boundary_coords = np.argwhere(boundary_map > 0)

            dilated = cv2.dilate(mask, kernel_large, iterations=1)
            ring_map = ((dilated - mask) > 0).astype(np.uint8)
            ring_coords = np.argwhere(ring_map > 0)

            # 正样本：多个点/短线，优先边界
            num_positive = random.randint(1, 4)
            for _ in range(num_positive):
                if random.random() < 0.6 and len(boundary_coords) > 0:
                    cy, cx = boundary_coords[random.randrange(len(boundary_coords))]
                else:
                    cy, cx = positive_points_coords[random.randrange(len(positive_points_coords))]

                brush_radius = random.randint(2, 4)
                cv2.circle(pos_scribble_map, (int(cx), int(cy)), brush_radius, 1, thickness=-1)

                # 概率生成短线
                if random.random() < 0.4:
                    steps = random.randint(2, 4)
                    px, py = int(cx), int(cy)
                    for _ in range(steps):
                        for _ in range(10):
                            dx = random.randint(-brush_radius*2, brush_radius*2)
                            dy = random.randint(-brush_radius*2, brush_radius*2)
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 1:
                                cv2.line(pos_scribble_map, (px, py), (nx, ny), 1, thickness=max(1, brush_radius))
                                px, py = nx, ny
                                break

            # 负样本：靠近边界的背景区域
            background_coords = np.argwhere(mask == 0)
            candidate_neg = ring_coords if len(ring_coords) > 0 else background_coords
            num_negative = random.randint(1, 3) if len(candidate_neg) > 0 else 0
            for _ in range(num_negative):
                ny, nx = candidate_neg[random.randrange(len(candidate_neg))]
                brush_radius = random.randint(2, 4)
                cv2.circle(neg_scribble_map, (int(nx), int(ny)), brush_radius, 1, thickness=-1)
                if random.random() < 0.3:
                    steps = random.randint(2, 4)
                    px, py = int(nx), int(ny)
                    for _ in range(steps):
                        for _ in range(10):
                            dx = random.randint(-brush_radius*2, brush_radius*2)
                            dy = random.randint(-brush_radius*2, brush_radius*2)
                            sx, sy = px + dx, py + dy
                            if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] == 0:
                                cv2.line(neg_scribble_map, (px, py), (sx, sy), 1, thickness=max(1, brush_radius))
                                px, py = sx, sy
                                break

        pos_scribble_map = (pos_scribble_map > 0).astype(np.uint8)
        neg_scribble_map = (neg_scribble_map > 0).astype(np.uint8)

        # 预处理原图 - 使用与推理时完全一致的方式
        image_tensor = self.image_transform(image)
        # 应用与推理时相同的归一化
        image_tensor = im_normalization(image_tensor)

        # 调整掩码和点击图尺寸以匹配图像大小
        mask_resized = np.array(Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST))
        pos_scribble_resized = np.array(Image.fromarray(pos_scribble_map).resize((self.size, self.size), Image.NEAREST))
        neg_scribble_resized = np.array(Image.fromarray(neg_scribble_map).resize((self.size, self.size), Image.NEAREST))

        # 将掩码和点击图转换为Tensor (保持0/1数值)
        gt_mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()
        pos_scribble_tensor = torch.from_numpy(pos_scribble_resized).unsqueeze(0).float()
        neg_scribble_tensor = torch.from_numpy(neg_scribble_resized).unsqueeze(0).float()

        # 模拟上一帧的掩码，对于第一帧交互，我们使用全零
        prev_mask_tensor = torch.zeros_like(gt_mask_tensor)
        
        # 拼接成6通道输入，完全模拟推理时的输入格式
        # 3 (image) + 1 (prev_mask) + 2 (pos_scribble + neg_scribble)
        Rs_tensor = torch.cat([pos_scribble_tensor, neg_scribble_tensor], dim=0)
        input_tensor = torch.cat([
            image_tensor,
            prev_mask_tensor,
            Rs_tensor
        ], dim=0)

        # 应用与推理时相同的padding
        input_tensor_padded, _ = pad_divide_by(input_tensor.unsqueeze(0), 16)
        gt_mask_padded, _ = pad_divide_by(gt_mask_tensor.unsqueeze(0), 16)

        return input_tensor_padded.squeeze(0), gt_mask_padded.squeeze(0)

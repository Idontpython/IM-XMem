import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# --- 路径修复 ---
# 这个代码块解决了相对导入的问题
# 它将项目的 'code' 文件夹添加到了Python的搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置(s2m_trainer)向上导航三次到达 'code' 目录
code_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(code_dir)
# --- 路径修复结束 ---


# 导入我们项目中的S2M模型结构和我们刚刚创建的数据集类 (使用绝对路径导入)
from inference.interact.s2m.s2m_network import deeplabv3plus_resnet50
from inference.interact.s2m_trainer.s2m_dataset import S2MDataset

# --- 损失函数定义 ---
# Dice Loss可以更好地处理类别不平衡问题，对分割任务很有效
def dice_loss(pred, target, smooth = 1.):
    # 注意：pred已经在调用前经过sigmoid，这里不需要再应用
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    a_sum = torch.sum(iflat * iflat)
    b_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (a_sum + b_sum + smooth))

# --- 主训练函数 ---
def train(args):
    # 1. 设置设备 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型并从原始S2M权重开始微调
    # num_classes=1 表示我们只做一个二分类：前景 vs 背景
    model = deeplabv3plus_resnet50(num_classes=1, pretrained_backbone=False)
    
    # 加载原始S2M权重作为初始化，而不是随机初始化
    try:
        original_s2m_path = os.path.join(args.root_path, 'code', 'saves', 's2m.pth')
        if os.path.exists(original_s2m_path):
            print(f"Loading original S2M weights from: {original_s2m_path}")
            original_weights = torch.load(original_s2m_path, map_location=device)
            model.load_state_dict(original_weights)
            print("✓ Successfully loaded original S2M weights as initialization")
        else:
            print("⚠ Original S2M weights not found, using random initialization")
    except Exception as e:
        print(f"⚠ Could not load original weights: {e}, using random initialization")
    
    model.to(device)

    # 3. 准备数据集
    train_dataset = S2MDataset(root_path=args.root_path, data_file=args.data_file, size=args.size)
    # DataLoader负责批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 4. 设置优化器和损失函数
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # 我们使用二元交叉熵和Dice Loss的组合，这是分割任务的常用策略
    bce_loss = nn.BCEWithLogitsLoss()

    # 5. 开始训练循环
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # 使用tqdm来显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for inputs, masks in pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            
            # 应用与推理时相同的后处理：sigmoid
            outputs_sigmoid = torch.sigmoid(outputs)

            # 计算损失 - 输出已过sigmoid
            loss_bce = torch.nn.functional.binary_cross_entropy(outputs_sigmoid, masks)
            loss_dice = dice_loss(outputs_sigmoid, masks)
            loss = loss_bce + loss_dice

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # 更新进度条信息
            pbar.set_postfix({'Loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        # 6. 保存模型
        # 每隔一定轮次就保存一次模型
        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            save_file = os.path.join(args.save_path, f's2m_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_file)
            print(f"Model saved to {save_file}")

    print("Training finished!")
    # 保存最终的模型
    final_save_file = os.path.join(args.save_path, 's2m_final.pth')
    torch.save(model.state_dict(), final_save_file)
    print(f"Final model saved to {final_save_file}")


# --- 主函数入口 ---
if __name__ == '__main__':
    # 使用argparse来管理命令行参数，方便调整超参数
    parser = argparse.ArgumentParser(description="Train S2M model")
    
    # 获取当前文件所在的目录，从而推断出项目根目录
    # /home/xushutan/IMXmem/code/inference/interact/s2m_trainer/train_s2m.py -> /home/xushutan/IMXmem
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_root_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    
    parser.add_argument('--root_path', type=str, default=default_root_path, help='Project root directory.')
    parser.add_argument('--data_file', type=str, default='datasets/train.txt', help='Path to the training data list file, relative to root_path.')
    parser.add_argument('--save_path', type=str, default='saves', help='Directory to save trained models, relative to root_path.')
    
    parser.add_argument('--size', type=int, default=384, help='Image size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (reduced for fine-tuning).')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training (smaller for stability).')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (smaller for fine-tuning).')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs.')

    args = parser.parse_args()
    
    # 运行训练函数
    train(args)

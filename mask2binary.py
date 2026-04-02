import cv2
import os

# 设置输入和输出文件夹路径
input_folder = ''  # 替换为实际路径
output_folder = '' # 输出路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的每个PNG文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 读取彩色掩码图像
        mask = cv2.imread(os.path.join(input_folder, filename))

        # 将图像转换为灰度图像
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 将灰度图像二值化
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

        # 保存二值化后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_mask)

print("所有掩码图像已转换并保存！")


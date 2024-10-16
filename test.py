import h5py
import numpy as np
from PIL import Image

import config


def load_and_convert_mask(mat_file, key='mask'):
    # 读取 .mat 文件
    with h5py.File(mat_file, 'r') as f:
        mask_data = f[key][...]  # 使用提供的键读取数据

    # 确保数据为 numpy 数组
    mask_array = np.array(mask_data)

    # 将 mask 数据归一化到 [0, 255] 范围
    mask_array = (mask_array - np.min(mask_array)) / (np.max(mask_array) - np.min(mask_array)) * 255
    mask_array = mask_array.astype(np.uint8)  # 转换为无符号8位整数
    mat_array = np.expand_dims(mask_array, axis=0)  # 添加通道维度
    print(mask_array.shape)
    # 创建灰度图像
    gray_image = Image.fromarray(mask_array, mode='L')

    return gray_image

# 使用示例
if __name__ == "__main__":
    mat_file1 = 'self_data/train_input/1.mat'  # 替换为你的 .mat 文件路径
    gray_image1 = load_and_convert_mask(mat_file1)

    # 保存或显示图像
    # gray_image1.save('mask_gray_image1.png')  # 保存为PNG格式
    # gray_image1.show()  # 显示图像

    # mat_file2 = 'self_data/train_output/1.mat'  # 替换为你的 .mat 文件路径
    # gray_image2 = load_and_convert_mask(mat_file2,key=config.output_key)

    # 保存或显示图像
    # gray_image2.save('mask_gray_image2.png')  # 保存为PNG格式
    # gray_image2.show()  # 显示图像

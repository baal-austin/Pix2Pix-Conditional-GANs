import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import torchvision.transforms as transforms

class MatDataset(Dataset):
    def __init__(self, mat_dir, label_dir, data_key, label_key, transform=None):
        self.mat_dir = mat_dir
        self.label_dir = label_dir
        self.mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
        self.transform = transform
        self.data_key = data_key
        self.label_key = label_key  # 添加标签键

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # 读取输入数据
        mat_path = os.path.join(self.mat_dir, self.mat_files[idx])
        with h5py.File(mat_path, 'r') as f:
            mat_data = f[self.data_key][:]  # 使用实际的键名读取数据

        # 将数据转换为PyTorch的张量
        data_tensor = torch.tensor(mat_data, dtype=torch.float32)
        data_tensor = data_tensor.unsqueeze(0)  # 增加通道维度 (1, 500, 500)

        # 读取对应的标签
        label_file = os.path.join(self.label_dir, self.mat_files[idx])  # 标签文件路径
        with h5py.File(label_file, 'r') as f:
            label = f[self.label_key][:]  # 使用 'train_output' 作为键名读取标签

        label_tensor = torch.tensor(label, dtype=torch.float32)  # 将标签转换为张量

        # 如果需要进行转换操作
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor  # 返回数据和标签

# 设置转换器
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))  # 对单通道进行归一化
])

# 加载数据集
mat_dir = 'self_data/train_input'
label_dir = 'self_data/train_output'  # 标签目录
data_key = 'mask'  # 对应.mat文件中的数据键（请确保这是实际存在的键）
label_key = 'mask_extend'  # 标签键
batch_size = 32

dataset = MatDataset(mat_dir, label_dir, data_key, label_key, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 测试DataLoader
# for batch in data_loader:
#     data, labels = batch
#     print("Batch data shape:", data.shape)  # 应该打印出 (32, 1, 500, 500)
#     print("Batch labels shape:", labels.shape)  # 标签形状应为与数据相匹配

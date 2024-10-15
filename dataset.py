import cv2
import numpy as np
import os
import config
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
import h5py
import torch
# from torch.utils.data import Dataset, DataLoader
class MapDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]
        target_image = image[:,600:,:]
        augmentations = config.both_transform(image=input_image,image0=target_image)
        input_image = augmentations['image']
        target_image = augmentations['image0']
        # 之后对 target_image 进行调整
        # target_image = cv2.resize(target_image, (1000, 1000))  # 将目标图像调整为1000x1000
        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_mask(image=target_image)['image']
        return input_image, target_image


class MatDataset1(Dataset):
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
        label_tensor = label_tensor.unsqueeze(0)  # 增加通道维度 (1, 500, 500)
        # 如果需要进行转换操作
        if self.transform:
            data_tensor = self.transform(data_tensor)
            label_tensor = self.transform(label_tensor)
        return data_tensor, label_tensor  # 返回数据和标签

if __name__ == "__main__":
    dataset = MapDataset(config.TRAIN_DIR)
    # dataset = MatDataset1(config.input_dir, config.output_dir, config.input_key, config.output_key)
    # print(dataset)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # save_image(x, "x.png")
        # save_image(y, "y.png")
        # import sys

        # sys.exit()

from albumentations.core.composition import Compose
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = "maps\\train"
VAL_DIR = "maps\\val"
input_dir = "self_data\\train_input"
output_dir = "self_data\\train_output"
input_key = 'mask'  # 对应.mat文件中的数据键（请确保这是实际存在的键）
output_key = 'mask_extend'  # 标签键
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 1000
CHANNEL_IMG = 1
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=1000,height=1000)],
    additional_targets={'image0': 'image'}
)
transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5],std=[0.5],max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5],std=[0.5],max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

test_only = A.Compose(
    [
        A.Resize(width=1000,height=1000),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
        ToTensorV2(),
    ]
)
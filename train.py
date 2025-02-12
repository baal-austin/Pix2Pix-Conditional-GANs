import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from utils import save_checkpoint, save_some_examples, load_checkpoint
import config
from generator import Generator
from discriminator import Discriminator


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        #train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            print('x', x.shape)
            print('y_fake', y_fake.shape)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    # 初始化生成器和判别器
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)

    # 初始化优化器
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # 损失函数
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # 检查是否需要加载模型
    if config.LOAD_MODEL:
        # 加载生成器和判别器的权重
        print("Loading model weights from config paths...")
        checkpoint_gen = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
        checkpoint_disc = torch.load(config.CHECKPOINT_DISC, map_location=config.DEVICE)

        # 应用加载的权重到生成器和判别器
        # ------------------------------------
        # 定义要忽略的层的名称
        ignore_layer_names = [
            'initial_down.0.weight',
            'final_up.0.weight',
            'final_up.0.bias'
        ]

        # 过滤掉不匹配的层
        for key in list(checkpoint_gen["state_dict"].keys()):
            if any(ignore_name in key for ignore_name in ignore_layer_names):
                del checkpoint_gen["state_dict"][key]
        # ------------------------------------
        ignore_layer_name = 'initial.0.weight'

        for key in list(checkpoint_disc["state_dict"].keys()):
            if ignore_layer_name in key:
                del checkpoint_disc["state_dict"][key]
        # ------------------------------------

        gen.load_state_dict(checkpoint_gen["state_dict"], strict=False)
        disc.load_state_dict(checkpoint_disc["state_dict"], strict=False)

        # 如果需要，恢复优化器的状态
        opt_gen.load_state_dict(checkpoint_gen["optimizer"])
        opt_disc.load_state_dict(checkpoint_disc["optimizer"])

    # 加载数据集
    # train_dataset = dataset.MapDataset(config.TRAIN_DIR)
    train_dataset = dataset.MatDataset1(config.input_dir, config.output_dir, config.input_key, config.output_key)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # val_dataset = dataset.MapDataset(config.VAL_DIR)
    val_dataset = dataset.MatDataset1(config.input_dir, config.output_dir, config.input_key, config.output_key)
    # val_dataset = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    # num_workers=config.NUM_WORKERS)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        # 定期保存模型
        if config.SAVE_MODEL and (epoch + 1) % 10 == 0:
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)

        # 生成并保存一些样本
        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()

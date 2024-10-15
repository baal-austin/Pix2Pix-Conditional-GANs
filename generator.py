import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm2d

class Block(nn.Module):
    def __init__(self,in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super(Block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='reflect')
            if down
            else
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self,in_channels=3,features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,3,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False)
        self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False)
        self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False)
        self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode='reflect'),
            nn.ReLU()
        )

        self.up1 = Block(features*8,features*8,down=False,act='relu',use_dropout=True)
        self.up2 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up3 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=True)
        self.up4 = Block(features*8*2,features*8,down=False,act='relu',use_dropout=False)
        self.up5 = Block(features*8*2,features*4,down=False,act='relu',use_dropout=False)
        self.up6 = Block(features*4*2,features*2,down=False,act='relu',use_dropout=False)
        self.up7 = Block(features*2*2,features,down=False,act='relu',use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        import torch.nn.functional as F

        # 下采样阶段
        print('x shape', x.shape)
        d1 = self.initial_down(x)
        print('d1.shape', d1.shape)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # 上采样阶段
        up1 = self.up1(bottleneck)

        # 确保 d7 的尺寸与 up1 一致
        if up1.shape[2:] != d7.shape[2:]:
            d7 = F.interpolate(d7, size=up1.shape[2:], mode='bilinear', align_corners=True)

        up2 = self.up2(torch.cat([up1, d7], dim=1))

        # 打印形状用于调试
        print("up2 shape:", up2.shape)
        print("d6 shape:", d6.shape)

        # 确保 up2 的尺寸与 d6 一致
        if up2.shape[2:] != d6.shape[2:]:
            d6 = F.interpolate(d6, size=up2.shape[2:], mode='bilinear', align_corners=True)

        up3 = self.up3(torch.cat([up2, d6], dim=1))

        # 打印形状用于调试
        print("up3 shape:", up3.shape)
        print("d5 shape:", d5.shape)

        # 确保 up3 的尺寸与 d5 一致
        if up3.shape[2:] != d5.shape[2:]:
            d5 = F.interpolate(d5, size=up3.shape[2:], mode='bilinear', align_corners=True)

        up4 = self.up4(torch.cat([up3, d5], dim=1))

        # 打印形状用于调试
        print("up4 shape:", up4.shape)
        print("d4 shape:", d4.shape)

        # 确保 up4 的尺寸与 d4 一致
        if up4.shape[2:] != d4.shape[2:]:
            d4 = F.interpolate(d4, size=up4.shape[2:], mode='bilinear', align_corners=True)

        up5 = self.up5(torch.cat([up4, d4], dim=1))

        # 打印形状用于调试
        print("up5 shape:", up5.shape)
        print("d3 shape:", d3.shape)

        # 确保 up5 的尺寸与 d3 一致
        if up5.shape[2:] != d3.shape[2:]:
            d3 = F.interpolate(d3, size=up5.shape[2:], mode='bilinear', align_corners=True)

        up6 = self.up6(torch.cat([up5, d3], dim=1))

        # 打印形状用于调试
        print("up6 shape:", up6.shape)
        print("d2 shape:", d2.shape)

        # 确保 up6 的尺寸与 d2 一致
        if up6.shape[2:] != d2.shape[2:]:
            d2 = F.interpolate(d2, size=up6.shape[2:], mode='bilinear', align_corners=True)

        up7 = self.up7(torch.cat([up6, d2], dim=1))

        # 打印形状用于调试
        print("up7 shape:", up7.shape)
        print("d1 shape:", d1.shape)

        # 确保 up7 的尺寸与 d1 一致
        if up7.shape[2:] != d1.shape[2:]:
            # d1 = F.interpolate(d1, size=up7.shape[2:], mode='bilinear', align_corners=True)
            up7 = F.interpolate(up7, size=d1.shape[2:], mode='bilinear', align_corners=True)

        #---
        ret = self.final_up(torch.cat([up7, d1], dim=1))
        print('up7', up7.shape)
        print('d1', d1.shape)
        print('ret', ret.shape)
        return ret


def test():
    x = torch.randn((1, 3, 500, 500))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
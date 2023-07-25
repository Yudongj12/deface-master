
# 定义 MobileNetV1-0.25模型：用于实现 1000类分类

import torch.nn as nn


# 定义 常规卷积块：3×3卷积 + 批归一化 + ReLU
def conv_bn(inp, oup, stride = 1, leaky = 0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


# 定义 深度可分离卷积块：3×3深度卷积 + 批归一化 + ReLU + 1×1卷积 + 批归一化 + ReLU
def conv_dw(inp, oup, stride = 1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


# 定义 MobileNetV1-0.25模型
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            conv_bn(3, 8, 2, leaky = 0.1),
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),

            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),

            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
            conv_dw(128, 128, 1), 
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), 
            conv_dw(256, 256, 1),
        )
        # 自适应平均池化：池化后的尺寸=1×1，通道数不变
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        # 全连接层：实现 1000类分类（有些诡异，输入通道数竟然较小！）
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

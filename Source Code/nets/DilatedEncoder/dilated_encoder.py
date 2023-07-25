
# 定义 瓶颈层DilatedEncoder
# 构成：Projector块（1×1卷积 + 3×3卷积） + 4个残差块（不同的扩展卷积倍数）
# 说明：Projector块 不改变 特征图尺寸，只改变 特征图通道数；残差块 不改变 特征图尺寸&通道数

import torch.nn as nn
from nets.DilatedEncoder.conv import Conv # 自定义基础卷积层Conv：含 普通卷积 + 批归一化 + 激活函数处理
# from utils import weight_init


# 残差结构 类（输入通道数&尺寸 = 输出通道数&尺寸）
class Bottleneck(nn.Module):
    # in_dim 输入通道数；dilation 3×3卷积步长；expand_ratio 缩放比例，决定 中间通道数；act_type 激活函数，默认 ReLU
    def __init__(self, 
                 in_dim, 
                 dilation=1, 
                 expand_ratio=0.25,
                 act_type='relu'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            # 1×1卷积层：降维
            # 使用 自定义基础卷积层Conv：含 普通卷积 + 批归一化 + 激活函数处理
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            # 3×3卷积层
            # 说明：p-paddle 边缘填充；d-dilation 扩展卷积，使卷积核扩展拉宽（但元素数不变），不影响输出特征
            # 注意：此时，步长=1，不会改变 特征图尺寸！！！
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type),
            # 1×1卷积层：升维
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )

    def forward(self, x):
        # 首尾相加（残差）
        return x + self.branch(x)


# DilatedEncoder类：用于声明 DilatedEncoder
# 输入声明：in_dim 输入特征图（来自Backbone）の通道数；out_dim 输出特征图の通道数；expand_ratio 缩放比例，决定 残差块の中间通道数；
#         dilation_list 扩展卷积倍数（对应4个残差块）；act_type 激活函数类型
class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio=0.25,
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu'):
        super(DilatedEncoder, self).__init__()

        # Projector块：1×1卷积 + 3×3卷积
        # 说明：不改变 特征图尺寸，只改变 特征图通道数
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None)
        )

        # 4个残差块（扩展卷积倍数不同）
        # 说明：不改变 特征图尺寸&通道数
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(in_dim=out_dim, 
                                       dilation=d, 
                                       expand_ratio=expand_ratio, 
                                       act_type=act_type))
        self.encoders = nn.Sequential(*encoders)

        # 初始化 权重参数
        self._init_weight()

    # 定义 函数：初始化 权重参数
    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                c2_xavier_fill(m)
                c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 前向计算
    def forward(self, x):
        # Projector 操作
        x = self.projector(x)
        # 残差 操作
        x = self.encoders(x)

        return x


# 定义 函数：部分权重 初始化
def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# 定义 改进RetinaFace模型：此处需确定 是否加载 预训练MobileNet模型
# 模型输出：(bbox_regressions, classifications, ldm_regressions) 人脸边框 + 分类 + 面部标记点
# 个人补充：添加 MobileNetV3-Large/Small 骨干网络（加载 官方示例）

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from nets.layers import conv_bn1X1, conv_bn
from torchvision import models

from nets.mobilenet025 import MobileNetV1
from nets.mobilenet_v2 import mobilenet_v2

from nets.DilatedEncoder.dilated_encoder import DilatedEncoder # 瓶颈层DilatedEncoder

from collections import OrderedDict


# Head输出端
# 输入声明：head_dim 输入特征图の通道数；num_anchors 锚类型数，默认 5种
class DecoupledHead(nn.Module):
    def __init__(self, head_dim=48, num_anchors=5):
        super().__init__()
        self.head_dim = head_dim

        # 分类预测层：3×3普通卷积；输出通道数 = 锚类型数 × 目标类别数（人脸 & 背景）
        self.cls_pred = nn.Conv2d(head_dim, 2 * num_anchors, kernel_size=(1,1), stride=1, padding=0)

        # 边框回归预测层：3×3普通卷积；输出通道数 = 锚类型数 × 4
        self.box_pred = nn.Sequential(
            # conv_bn1X1(head_dim, head_dim, stride=1, leaky=0.1),  # 通道数<=64，leaky=0.1；否则 leaky=0
            nn.Conv2d(head_dim, 4 * num_anchors, kernel_size=(1, 1), stride=1, padding=0),
        )

        # 面部标记点回归预测层：3×3普通卷积；输出通道数 = 锚类型数 × 4(Only 2 eye + 1 nose)
        self.landm_pred = nn.Sequential(
            # conv_bn1X1(head_dim, head_dim, stride=1, leaky=0.1), # 通道数<=64，leaky=0.1；否则 leaky=0
            nn.Conv2d(head_dim, 6 * num_anchors, kernel_size=(1,1), stride=1, padding=0)
        )

    # 前向计算流程
    def forward(self, x):

        # 计算 分类预测特征图 & 边框回归预测特征图 & 面部标记点回归预测特征图
        cls_pred = self.cls_pred(x)

        box_pred = self.box_pred(x)

        landm_pred = self.landm_pred(x)

        # 调整 尺寸 & 维度（方便训练）
        cls_pred = cls_pred.permute(0,2,3,1).contiguous() # 维度调换
        box_pred = box_pred.permute(0,2,3,1).contiguous() # 维度调换
        landm_pred = landm_pred.permute(0,2,3,1).contiguous() # 维度调换

        # 输出
        # 分类结果：批大小 × 总像素数 × 2分类值
        # 边框回归结果：批大小 × 总像素数 × 4（中心点坐标+宽高<回归值>）
        # 面部标记点回归结果：批大小 × 总像素数 × 6（3点坐标<回归值>, only 2 eye + 1 nose）
        return cls_pred.view(cls_pred.shape[0], -1, 2), box_pred.view(box_pred.shape[0], -1, 4), landm_pred.view(landm_pred.shape[0], -1, 6)


# 定义 RetinaFace模型
class RetinaFace(nn.Module):
    def __init__(self, cfg = None, pretrained = False, mode = 'train'):
        super(RetinaFace,self).__init__()
        backbone = None

        # 骨干网络Backbone
        # -----------------------------------------------------------------------------------
        self.mark = 'origin'  # 骨干网络选择标志：为了区分 Backbone（加载方式不同）

        # 选择 骨干网络backbone：MobileNetV1-0.25、MobileNetV2、MobileNetV3-Large、MobileNetV3-Small、ResNet18
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            self.mark = 'v1_0.25'
            # 是否加载 预训练MobileNet模型：虽然默认 不加载，但实际使用时 加载！
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                # 预防措施：防止网络参数不完全一致 导致的 加载失败
                # ——————————————————————————————————————————————————
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
                # ——————————————————————————————————————————————————
        # 个人补充：MobileNetV2
        elif cfg['name'] == 'mobilenetv2':
            backbone = mobilenet_v2(pretrained=pretrained)
            self.mark = 'v2'
        # 个人补充：MobileNetV3-Large
        elif cfg['name'] == 'mobilenetv3large':
            backbone = models.mobilenet_v3_large(pretrained=pretrained)
            self.mark = 'v3_large'
        # 个人补充：MobileNetV3-Small
        elif cfg['name'] == 'mobilenetv3small':
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.mark = 'v3_small'
        # 个人补充：ResNet18
        elif cfg['name'] == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.mark = 'resnet18'

        # 获取backbone的 输出特征
        if cfg['name'] == 'mobilenet0.25':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])  # IntermediateLayerGetter 获取模型网络中 某些层输出的特征
        # MobileNetV2 & MobileNetV3-Large/Small (使用另一种获取方式)
        elif cfg['name'] == 'mobilenetv2' or cfg['name'] == 'mobilenetv3small' or cfg['name'] == 'mobilenetv3large':
            self.body = backbone.features
        # ResNet18
        elif cfg['name'] == 'resnet18':
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        # -----------------------------------------------------------------------------------

        # 特征融合部分
        # -----------------------------------------------------------------------------------
        # 3个特征图的 原始通道数
        if cfg['name'] == 'mobilenet0.25':
            in_channels_list = [64, 128, 256]
        elif cfg['name'] == 'mobilenetv2':
            in_channels_list = [32, 96, 320]
        elif cfg['name'] == 'mobilenetv3large':
            in_channels_list = [40, 112, 160]
        elif cfg['name'] == 'mobilenetv3small':
            in_channels_list = [24, 48, 96]
        elif cfg['name'] == 'resnet18':
            in_channels_list = [128, 256, 512]

        # 如果 输出通道数<=64，则 leaky = 0.1；否则 leaky = 0.0
        # self.BackDeal1 = conv_bn(in_channels_list[0], cfg['out_channel'], stride=2, leaky=0.1)  # 3×3卷积块
        # self.BackDeal1 = conv_bn1X1(in_channels_list[0], cfg['out_channel'], stride=1, leaky=0.1)  # 1×1卷积块
        self.BackDeal2 = conv_bn1X1(in_channels_list[1], cfg['out_channel'], stride=1, leaky=0.1)
        self.BackDeal3 = conv_bn1X1(in_channels_list[2], cfg['out_channel'], stride=1, leaky=0.1)
        # self.BackDeal11 = conv_bn1X1(cfg['out_channel'], cfg['out_channel'], stride=1, leaky=0.1)
        # self.BackDeal22 = conv_bn1X1(cfg['out_channel'], cfg['out_channel'], stride=1, leaky=0.1)
        # -----------------------------------------------------------------------------------

        # 瓶颈层DilatedEncoder
        # 输入声明：cfg['in_channel'] 输入特征图（来自Backbone）の通道数；cfg['out_channel'] 输出特征图の通道数
        #         expand_ratio 缩放比例，决定 残差块の中间通道数；dilation_list 扩展卷积倍数（对应4个残差块）；act_type 激活函数类型
        # C5 [2, 4, 6, 8]; C4 [4, 8, 12, 16]; C4-640 [4, 8, 12, 16]（小尺寸）
        self.neck = DilatedEncoder(cfg['in_channel'], cfg['out_channel'], expand_ratio=0.25,
                                   dilation_list=[2, 4, 6, 8], act_type='lrelu')  # conv.py get_activation() Change act_type; Choose lrelu

        # 输出端Head
        # 输入声明：head_dim 输入特征图の通道数；num_anchors 锚类型数，默认 5种
        self.head = DecoupledHead(head_dim=cfg['out_channel'], num_anchors=cfg['num_anchor'])

        # 网络模式：训练模式 & 测试模式，决定了 模型の输出
        self.mode = mode

    # 前向计算过程
    def forward(self, inputs):

        # 获得 Backbone输出特征图
        # MobileNetV1-0.25
        if self.mark == 'v1_0.25':
            out = self.body.forward(inputs)  # 3个特征图，通道数 64 128 256, 缩小 8 16 32倍
        # MobileNetV2
        elif self.mark == 'v2':
            out = OrderedDict()
            out[1] = self.body[:7](inputs)     # C3特征图，通道数  32, 缩小  8倍
            out[2] = self.body[7:14](out[1])   # C4特征图，通道数  96, 缩小 16倍
            out[3] = self.body[14:18](out[2])  # C5特征图，通道数 320, 缩小 32倍
        # MobileNetV3-Large
        elif self.mark == 'v3_large':
            out = OrderedDict()
            out[1] = self.body[:7](inputs)     # C3特征图，通道数  40, 缩小  8倍
            out[2] = self.body[7:13](out[1])   # C4特征图，通道数 112, 缩小 16倍
            out[3] = self.body[13:16](out[2])  # C5特征图，通道数 160, 缩小 32倍
        # MobileNetV3-Small
        elif self.mark == 'v3_small':
            out = OrderedDict()
            out[1] = self.body[:4](inputs)     # C3特征图，通道数  24, 缩小  8倍
            out[2] = self.body[4:9](out[1])    # C4特征图，通道数  48, 缩小 16倍
            out[3] = self.body[9:12](out[2])   # C5特征图，通道数  96, 缩小 32倍
        # ResNet18
        elif self.mark == 'resnet18':
            out = self.body.forward(inputs)  # 3个特征图，通道数 128 256 512, 缩小 8 16 32倍

        # 特征融合
        # -----------------------------------------------------------------------------------
        # out1 = self.BackDeal1(out[1])
        out2 = self.BackDeal2(out[2])
        out3 = self.BackDeal3(out[3])

        # Method1: C3 & C5 叠加至 C4
        # out2222 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest") + F.interpolate(out1, size=[out2.size(2), out2.size(3)], mode="nearest")

        # 方案一: C3 & C5 叠加至 C4(C3：步长=2的3×3卷积 代替 下采样)
        # out2222 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest") + out1

        # Method2: 参考PANet，上采样 + 横向连接 + 下采样 + 横向连接（3个1×1卷积+2个上采样+2个1×1卷积+1个下采样）
        # out22 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest")
        # out222 = self.BackDeal22(out22)
        # out11 = out1 + F.interpolate(out222, size=[out1.size(2), out1.size(3)], mode="nearest")
        # out111 = self.BackDeal11(out11)
        # out2222 = out222 + F.interpolate(out111, size=[out222.size(2), out222.size(3)], mode="nearest")

        # 方案二: 参考 PANet路径聚合网络，对C3、C4、C5特征 依次进行 上采样、横向连接、下采样、横向连接，生成D4特征图
        # out11 = out1 + F.interpolate(out2, size=[out1.size(2), out1.size(3)], mode="nearest") # Up-sample
        # out22 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest") # Up-sample
        # out2222 = out22 + F.interpolate(out11, size=[out22.size(2), out22.size(3)], mode="nearest") # Down-sample

        # 方案二: 参考 PANet路径聚合网络，对C3、C4、C5特征 依次进行 上采样、横向连接、下采样、横向连接，生成D4特征图(增加 2个 1×1卷积)
        # out11 = out1 + F.interpolate(out2, size=[out1.size(2), out1.size(3)], mode="nearest")
        # out22 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest")
        # out111 = self.BackDeal11(out11)
        # out222 = self.BackDeal22(out22)
        # out2222 = out222 + F.interpolate(out111, size=[out222.size(2), out222.size(3)], mode="nearest")

        # Method3: 参考FPN，C4 + C5上采样（2个1×1卷积+1个上采样）
        out2222 = out2 + F.interpolate(out3, size=[out2.size(2), out2.size(3)], mode="nearest")  # Up-sample

        # Method4: 仅C4
        # out2222 = out2
        # -----------------------------------------------------------------------------------

        # 使用 瓶颈层DilatedEncoder，优化特征图
        out_final = self.neck(out2222)

        # Head输出端 操作
        cls_pred, box_pred, landm_pred = self.head(out_final)

        # 根据 网络模式mode，确定 输出格式（如果为 预测模式，需要使用Softmax进行 概率归一）
        if self.mode == 'train':
            output = (box_pred, cls_pred, landm_pred)
        else:
            output = (box_pred, F.softmax(cls_pred, dim=-1), landm_pred)
        return output

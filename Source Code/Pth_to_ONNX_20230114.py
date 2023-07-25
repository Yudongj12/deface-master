
# My Face Detection 模型导出测试：Pth -→ ONNX

import os
import torch
import torch.nn as nn
import onnx

from utils.config import cfg_mnet # 模型配置参数：MobileNetV1-0.25
from nets.retinaface import RetinaFace # 模型结构
from utils.anchors import Anchors # 锚(含 尺寸 & 位置分布)


# (待定)含锚の模型：输出 预测值 & 原始锚框
class CompleteModel(nn.Module):
    def __init__(self, backbone):
        super(CompleteModel, self).__init__()
        self.backbone = backbone

    def forward(self, inputs):
        # anchors = Anchors(cfg=cfg_mnet, image_size=(640, 640)).get_anchors()
        # print(anchors.shape)
        outputs = self.backbone(inputs)

        # 输出顺序：分类值-2、边框回归值-4、标记点回归值-6
        out = torch.cat([outputs[1], outputs[0], outputs[2]], dim=-1)
        return torch.cat([outputs[1], outputs[0], outputs[2]], dim=-1)


# 主程序
if __name__ == '__main__':
    # 加载模型结构
    net = RetinaFace(cfg=cfg_mnet, mode='eval').eval()

    # 读取 已训练模型
    print('Loading weights into state dict...')
    state_dict = torch.load('logs/V1_M3_12.03_Loss6.5066.pth')
    net.load_state_dict(state_dict)
    net = net.eval()
    print('Finished!')

    # 输入图像尺寸
    Virture_Input = torch.randn(1, 3, 640, 640)
    # 模型输入 命名
    input_names = ['input_image']
    # 模型输出 命名：边框回归值，人脸/背景 分类值，面部标记点回归值
    output_names = ['bbox_regressions', 'classifications', 'ldm_regressions']

    # # 导出 ONNX模型文件：dynamic_axes 动态参数设置
    # torch.onnx.export(net,
    #                   Virture_Input,
    #                   'My_FaceDetection.onnx',
    #                   dynamic_axes={'input_image': {0: 'batch', 2: 'H', 3: 'W'},
    #                                 'bbox_regressions': {0: 'batch', 1: 'S'},
    #                                 'classifications': {0: 'batch', 1: 'S'},
    #                                 'ldm_regressions': {0: 'batch', 1: 'S'}},
    #                   input_names=input_names,
    #                   output_names=output_names,
    #                   opset_version=11)
    #
    # # 检验 ONNX模型文件 准确性
    # onnx_model = onnx.load(r"My_FaceDetection.onnx")
    # try:
    #     onnx.checker.check_model(onnx_model)
    # except Exception:
    #     print("Model incorrect")
    # else:
    #     print("Model correct")

    complete_model = CompleteModel(backbone=net)
    output_name = ['complete_model_output']

    torch.onnx.export(complete_model,
                      Virture_Input,
                      'My_FaceDetection.onnx',
                      input_names=input_names,
                      output_names=output_name,
                      opset_version=11
                      )

    # 检验 ONNX模型文件 准确性
    onnx_model = onnx.load(r"My_FaceDetection.onnx")
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")

    # torch.onnx.export(complete_model,
    #                   Virture_Input,
    #                   'complete_model.onnx',
    #                   dynamic_axes={'input_image': {0: 'batch', 2: 'H', 3: 'W'},
    #                                 'complete_model_output': {0: 'batch', 1: 'S'}},
    #                   input_names=input_names,
    #                   output_names=output_name,
    #                   opset_version=11
    #                   )























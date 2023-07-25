
# 获取 图像中 所有锚的位置信息（中心点坐标 + 宽&高）<归一化>
# 注意：存在 超出边界的锚，位于边缘的锚 超出 图像边界

from itertools import product as product
from math import ceil

import torch


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        # 锚的边长：[32, 64, 128, 256, 512]
        self.min_sizes = cfg['min_sizes']
        # 特征图相对于原图的 缩放倍数：16（计算方式=锚的步长）
        self.steps = cfg['steps']
        # False
        self.clip = cfg['clip']
        # 图像尺寸：640×640
        self.image_size = image_size
        # 3个不同尺度特征图的 高&宽：80×80、40×40、20×20
        self.feature_maps = [ceil(self.image_size[0] / self.steps), ceil(self.image_size[1] / self.steps)]

    def get_anchors(self):

        anchors = [] # 用于存储 所有锚

        # 存储 特征图の所有锚（中心点坐标 + 宽&高）<归一化>
        # 1个特征图 对应的 锚边长（5个，如 [32, 64, 128, 256, 512]）
        min_sizes = self.min_sizes

        # 遍历 特征图の所有锚
        # i, j 当前特征图的 锚点坐标（从0递增）
        for i, j in product(range(self.feature_maps[0]), range(self.feature_maps[1])):
            for min_size in min_sizes:
                # 1个锚的 宽&高
                s_kx = min_size / self.image_size[1]
                s_ky = min_size / self.image_size[0]

                # 1个锚的 中心点坐标（归一化）
                # steps 锚的步长；image_size 图像尺寸640；j+0.5/i+0.5 映射至 锚的中心位置
                dense_cx = [x * self.steps / self.image_size[1] for x in [j + 0.5]]
                dense_cy = [y * self.steps / self.image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)

        return output

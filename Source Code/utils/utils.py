
# 部分函数（训练阶段 & 预测阶段 均有）

import cv2
import numpy as np


# 定义函数：对输入图像进行 无失真尺寸调节（预测阶段使用）
def letterbox_image(image, size):
    ih, iw, _   = np.shape(image)
    w, h        = size
    scale       = min(w/iw, h/ih)
    nw          = int(iw*scale)
    nh          = int(ih*scale)

    image       = cv2.resize(image, (nw, nh))
    new_image   = np.ones([size[1], size[0], 3]) * 128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image
    

# 定义函数：获得 学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 定义函数：对图像进行 标准化处理
def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image

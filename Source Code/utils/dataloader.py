
# 训练集 数据生成器：读取 原始图像 & 人脸标注信息，对原始图像进行 随机调整（尺寸统一为840×840，已扭曲！）、标准化处理
# DataGenerator输出声明：调整后的 图像 & 人脸标注信息
# 作用方式：train.py，调用 DataGenerator 读取 训练集，并进行 预处理操作

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.utils import preprocess_input


class DataGenerator(data.Dataset):
    def __init__(self, txt_path, img_size):
        self.img_size = img_size
        self.txt_path = txt_path

        # 调用函数process_labels：根据 训练集标注文件label.txt，获取 所有图像路径imgs_path & 对应的人脸标注信息words
        self.imgs_path, self.words = self.process_labels()

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        # 打开图像，获取对应的人脸标注信息
        img = Image.open(self.imgs_path[index])
        labels = self.words[index] # 1张图像对应的 所有人脸标注信息
        annotations = np.zeros((0, 15)) # 用于存放 1张图像对应的 所有人脸标注信息（已变更内容）

        if len(labels) == 0:
            return img, annotations

        # 遍历1张图像对应的 1个人脸标注信息（对应label.txt的1行），存入annotation，并添加至annotations
        # 每行原始内容：左上x，左上y，宽，高，点1x，点1y，blur模糊？，点2x，点2y，expression表情？，点3x，点3y，illumination光照？
        #            点4x，点4y，occlusion遮挡程度？，点5x，点5y，pose姿态？，invalid有效性？
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox 人脸真实框的位置（左上点 + 右下点）
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            # landmarks 人脸标记点的位置（5个点）
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x（跳过第6个元素，因为该元素不代表坐标值！）
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            # 该真实人脸边框 是否含 面部标记点：1 含；-1 不含
            # 原因：有些人脸太过模糊/位置不正，无法标注面部点位
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations) # 用于存放 1张图像对应的 所有人脸标注信息（已变更内容）
        # 调用函数get_random_data，将图像尺寸修改为img_size×img_size（840×840），进行 随机缩放（扭曲图像！！），边缘补灰边，随机翻转，随机色域变换
        # img 随机调整后的图像；target 对应的 人脸标注信息
        img, target = self.get_random_data(img, target, [self.img_size,self.img_size])
        # 调用 preprocess_input函数，进行标准化处理
        img = np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1))
        return img, target

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    # 定义 图像预处理函数：将图像尺寸修改为img_size×img_size（1280×1280），缩放至 匹配1280×1280，边缘补灰边，随机翻转，随机色域变换
    def get_random_data(self, image, targes, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih = image.size
        h, w = input_shape # h=1280, w=1280
        box = targes

        # 对图像进行缩放 -----------------------------------------
        new_ar = iw/ih # Vital!!!!!!
        if new_ar < 1:
            nh = h
            nw = int(nh*new_ar)
        else:
            nw = w
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 对图像进行缩放并且进行长和宽的扭曲 ------------------------ Bad!!!!!!!!!!!!!!
        # new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        # scale = self.rand(0.25, 3.25)
        # if new_ar < 1:
        #     nh = int(scale * h)
        #     nw = int(nh * new_ar)
        # else:
        #     nw = int(scale * w)
        #     nh = int(nw / new_ar)
        # image = image.resize((nw, nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # image.show()
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2,4,6,8,10,12]] = box[:, [0,2,4,6,8,10,12]]*nw/iw + dx
            box[:, [1,3,5,7,9,11,13]] = box[:, [1,3,5,7,9,11,13]]*nh/ih + dy
            if flip: 
                box[:, [0,2,4,6,8,10,12]] = w - box[:, [2,0,6,4,8,12,10]]
                box[:, [5,7,9,11,13]]     = box[:, [7,5,9,13,11]]
            
            center_x = (box[:, 0] + box[:, 2])/2
            center_y = (box[:, 1] + box[:, 3])/2
        
            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

            box[:, 0:14][box[:, 0:14]<0] = 0
            box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
            box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h
            
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        box[:,4:-1][box[:,-1]==-1]=0
        box[:, [0,2,4,6,8,10,12]] /= w
        box[:, [1,3,5,7,9,11,13]] /= h
        box_data = box
        return image_data, box_data
        
    # 定义函数：根据 训练集标注文件label.txt，获取 所有图像路径imgs_path & 对应的人脸标注信息words
    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        # 遍历 label.txt的 每行内容，获取 图像路径 & 对应的人脸标注信息
        for line in lines:
            line = line.rstrip()
            # 如果该行为图像路径（含#），则存为 图像路径；否则存为 人脸标注信息
            if line.startswith('#'):
                # 判断：每经过1张图像，将 该图像的所有人脸标注信息labels 添加至 words，并清空labels
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:] # 仅获取 图像路径（去掉第0&1个元素）
                # 获取 完整路径（从程序根目录开始），如'data/widerface/train/images/0--Parade/0_Parade_marchingband_1_849.jpg'
                path = self.txt_path.replace('label.txt','images/') + path # 使用'images/'代替'label.txt'，并添加path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels) # 针对 最后一张图像（因为遍历完后，不再判断isFirst）
        return imgs_path, words

def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets

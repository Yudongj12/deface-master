
# 训练模型
# 默认配置：使用GPU；选择 MobileNetV1-0.25为 骨干网络backbone；加载 backboneの预训练权重（不加载 本地RetinaFace模型）；不冻结训练
#         150次迭代训练；SGD优化器；每5次迭代，保存1次模型文件
# 个人修改：不再记录&存储 损失值

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.retinaface import RetinaFace
from nets.retinaface_training import (MultiBoxLoss, get_lr_scheduler,
                                      set_optimizer_lr, weights_init)
from utils.anchors import Anchors
from utils.callbacks import LossHistory
from utils.config import cfg_mnet, cfg_mnetv2, cfg_mnetv3_large, cfg_mnetv3_small, cfg_resnet18
from utils.dataloader import DataGenerator, detection_collate
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    # 参数设置
    # ——————————————————————————————————————————————————
    # 是否使用GPU
    Cuda = True

    # 训练集の人脸标注文件label.txt
    training_dataset_path = 'data/widerface/train/label.txt'

    # 骨干网络backbone的选择：mobilenetV1-0.25、mobilenetv2、mobilenetv3large、mobilenetv3small、resnet18
    backbone = "mobilenetV3small"

    # 是否使用 骨干网络backboneの预训练权重（在模型构建的时候进行加载）
    # 注意：如果设置了model_path，则不加载 骨干网络の预训练权重，pretrained 无意义
    # 注意：如果不设置model_path，pretrained = True，此时仅加载 预训练好的骨干网络 开始训练
    # 注意：如果不设置model_path，pretrained = False，Freeze_Train = False，此时从头开始训练，且没有冻结backbone的过程
    pretrained = True
    # pretrained = False # 个人修改：断点续练

    # 本地 RetinaFace模型路径
    # 注意：如果 model_path = ''，不加载 本地模型
    # 注意：如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件；同时修改下方的 冻结阶段/解冻阶段 的参数，保证模型迭代的连续性
    # 注意：如果想要让模型从 预训练好的骨干网络 开始训练，则设置model_path = ''，pretrain = True，此时仅加载 骨干网络の预训练权重（初始化其余网络参数）
    # 注意：如果想要让模型从头开始训练，则设置model_path = ''，pretrain = False，Freeze_Train = False，此时从0开始训练，且没有冻结backbone的过程
    # 警告：一般来讲，网络从头开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    # 警告：如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    model_path = ""
    # model_path = "logs/1time-Total_Loss8.0155.pth" # 个人修改：断点续练

    # 训练分为两个阶段：冻结阶段 & 解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求
    # 冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练
    # 在此提供若干参数设置建议，请根据自己的需求进行灵活调整：
    # （一）从主干网络的预训练权重开始训练：
    # Adam：
    # Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0（冻结）
    # Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0（不冻结）
    # SGD：
    # Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4（冻结）
    # Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4（不冻结）
    # 其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解
    # UnFreeze_Epoch可以在150-300之间调整，YOLOV5和YOLOX均推荐使用300。
    # Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    # （二）batch_size的设置：
    # 在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size
    # 受到BatchNorm层影响，batch_size最小为2，不能为1
    # 正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整

    # 冻结训练阶段 训练参数
    # 注意：此时backbone被冻结，仅对网络进行微调（占用的显存较小）
    # 初始迭代次数：其值可以大于Freeze_Epoch，如设置Init_Epoch = 60、Freeze_Epoch = 50，会跳过冻结阶段，直接从第60代开始，并调整对应的学习率（断点续练时使用）
    Init_Epoch = 0
    # Init_Epoch = 3 # 个人修改：断点续练
    # 冻结迭代次数：前50次迭代为 冻结阶段(Freeze_Train=False时失效)
    Freeze_Epoch = 0 # Not Use
    # 冻结训练阶段のbatch批大小（1个批包含的样本数）
    Freeze_batch_size = 64 # Not Use

    # 解冻/正常训练阶段 训练参数
    # 注意：此时不再冻结backbone，占用显存较大，网络的所有参数都会发生改变
    # 总迭代数
    UnFreeze_Epoch = 300 # Default 300
    # 解冻训练阶段のbatch批大小（1个批包含的样本数）：＜冻结阶段，因为此时需要训练更多参数（默认8, 可选64）
    Unfreeze_batch_size = 64 # 本地PC 16；远程服务器 128 256

    # Freeze_Train：是否进行冻结训练（默认 不冻结）
    Freeze_Train = False

    # 其它训练参数：学习率、优化器、学习率下降相关
    # 模型的最大学习率：使用Adam优化器时建议设置1e-3；使用SGD优化器时建议设置1e-2
    Init_lr = 1e-2 # default 1e-2
    # 模型的最小学习率，默认为最大学习率的0.01
    Min_lr = Init_lr * 0.01 # default Init_lr * 0.01

    # 优化器类型：可选 adam、sgd；默认 sgd
    optimizer_type = "sgd" # 2022.8.19 更改为 adam
    # 优化器内部的 momentum动量参数
    momentum = 0.937
    # 权重衰减：可防止过拟合（adam会导致weight_decay错误，使用adam时建议设置为0）
    weight_decay = 5e-4 # sgd 5e-4

    # 学习率下降方式：可选 'step'、'cos'
    lr_decay_type = 'cos' # default cos

    # save_period：多少个epoch保存一次权值，默认=1（每个世代都保存）
    save_period = 10

    # 权值与日志文件保存的文件夹
    save_dir = 'logs'

    # 多线程设置：0代表关闭多线程, 可选16 32
    # 注意：开启后会加快数据读取速度，但是会占用更多内存；在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度
    num_workers = 8 # 本地PC 2；远程服务器 16
    # ——————————————————————————————————————————————————

    # 加载 RetinaFace模型
    # ——————————————————————————————————————————————————
    # 选择 骨干网络backbone，根据 config.py 配置模型
    if backbone == "mobilenetV1-0.25":
        cfg = cfg_mnet
    elif backbone == "mobilenetv2":
        cfg = cfg_mnetv2
    elif backbone == "mobilenetv3large":
        cfg = cfg_mnetv3_large
    elif backbone == "mobilenetv3small":
        cfg = cfg_mnetv3_small
    elif backbone == "resnet18":
        cfg = cfg_resnet18
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    # 实例化模型
    model = RetinaFace(cfg=cfg, pretrained=pretrained)
    # 如果 模型未预训练，则初始化 模型权重
    if not pretrained:
        weights_init(model)

    # 加载 本地预训练模型
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 个人修改：不再记录并保存损失
    loss_history = 0
    # loss_history = LossHistory(save_dir, model, input_shape=(cfg['train_image_size'], cfg['train_image_size']))

    # 初始化 多任务损失：类别数=2<人脸/背景>、正负锚判断阈值、正负锚比例、计算权重、使用GPU？
    criterion = MultiBoxLoss(2, cfg['P_Thres'], cfg['N_Thres'], cfg['k_num'], cfg['NegPos_ratio'], cfg['variance'], Cuda)

    model_train = model.train() # 模型设为 训练模式
    # GPU加速设置
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 获取 图像中（3个特征图包含的） 所有锚的位置信息（中心点坐标 + 宽&高）<归一化>
    # 用途：判断 正负锚 + 计算 损失函数 + 人脸边框回归
    anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
    if Cuda:
        anchors = anchors.cuda()
    # ——————————————————————————————————————————————————

    if True:
        # 实例化 优化器、数据集
        # ——————————————————————————————————————————————————
        # 注意：冻结训练可以加快训练速度；提示OOM或者显存不足请调小Batch_size！！！
        UnFreeze_flag = False
        # 冻结 骨干网络backbone（主干特征提取部分）的权重参数！
        if Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = False
        # 如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        # 判断当前batch_size，自适应调整学习率
        nbs = 64 # default 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        # Init_lr_fit = 0.001 # 个人修改：再次训练
        # Min_lr_fit = 0.001 # 个人修改：再次训练

        # 优化器：根据optimizer_type选择优化器
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay}) # weight_decay 权重衰减=0.0005
        optimizer.add_param_group({"params": pg2})

        # 获得学习率下降的公式: 函数声明中 可修改 Warm-Up 迭代次数
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 调用 DataGenerator 读取 训练集，并进行 预处理操作
        # 输入声明：training_dataset_path 训练集标注文件label.txt文件路径；cfg['train_image_size'] 输入模型的图像尺寸（1个边）
        train_dataset = DataGenerator(training_dataset_path, cfg['train_image_size'])

        # gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        #                 drop_last=True, collate_fn=detection_collate)
        # 个人修改（提高GPU利用率）：persistent_workers=True
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate, persistent_workers=True)
        # batch批の数量 epoch_step = 训练图像总数÷批大小（向下取整）
        epoch_step = train_dataset.get_len() // batch_size
        # ——————————————————————————————————————————————————

        # 测试代码：保存模型，查看大小
        import os
        torch.save(model.state_dict(), os.path.join(save_dir, 'Test.pth'))

        # 训练模型：150次迭代，可选是否冻结主干
        # ——————————————————————————————————————————————————
        # Init_Epoch 起始迭代数（如果没有中断迭代训练，默认为0）；UnFreeze_Epoch 总迭代数
        # 提示：前Freeze_Epoch次为 冻结迭代，之后为 解冻/正常迭代
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            # 2022.8.4 重复加载 训练数据集
            # train_dataset = DataGenerator(training_dataset_path, cfg['train_image_size'])
            # gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
            #                  pin_memory=True,
            #                  drop_last=True, collate_fn=detection_collate, persistent_workers=True)

            # 如果模型有冻结学习部分，则解冻，并设置参数
            # 注意：如果 Freeze_Train=False（默认不冻结），该部分代码无用！！！
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # 判断当前batch_size，自适应调整学习率
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # 获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                # 解冻 backbone
                for param in model.body.parameters():
                    param.requires_grad = True
                # epoch_step 训练集的 总批数
                epoch_step = train_dataset.get_len() // batch_size
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=detection_collate)
                UnFreeze_flag = True

            # 调用 set_optimizer_lr函数，设置 优化器
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            # 调用 fit_one_epoch函数，进行 1次迭代训练
            fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen, UnFreeze_Epoch, anchors, cfg, Cuda, save_period, save_dir)
        # ——————————————————————————————————————————————————
        # 个人修改：不再记录并保存损失
        # loss_history.writer.close() # 关闭 损失文件保存（因为训练已结束）

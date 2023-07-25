
# 模型配置文件：定义 骨干网络backbone & 对应设置参数

# MobileNetV1-0.25
cfg_mnet = {
    'name': 'mobilenet0.25',
    'num_anchor': 6, # 5种锚
    'min_sizes': [8, 16, 32, 64, 128, 256], # 锚的边长：默认 5种锚 [32, 64, 128, 256, 512] [8, 16, 32, 64, 128, 256]
    'steps': 16, # 特征图缩放倍数（相对于原图）：16 （计算方式=锚的步长）
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}, # 输出特征层，单层表达 {'stage1': 1, 'stage2': 2, 'stage3': 3}; Method4 {'stage1': 1, 'stage2': 2}
    'P_Thres': 0.3, # 正样本 筛选阈值：滤除 IoU＜P_Thres的 正样本; default 0.15
    'N_Thres': 0.3, # 负样本 筛选阈值：滤除 IoU＞N_Thres的 负样本; default 0.35
    'k_num': 5, # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 10，总数为 20
    'NegPos_ratio': 10, # 负正锚比例：调整为 10倍（避免负锚过多，影响训练）
    'variance': [0.1, 0.2], # 用于计算 真实边框回归值&真实面部标记点回归值，有利于 计算损失
    'clip': False,
    'cls_weight': 1.0, # 分类损失       所占的比重
    'loc_weight': 2.0, # 边框回归损失    所占的比重
    'landm_weight': 1.0, # 标记点回归损失  所占的比重
    'train_image_size': 640, # 缩减为 640
    'in_channel': 64, # BackBone输出特征图の通道数 = DilatedEncoder输入特征图の通道数
    'out_channel': 64 # 经过DilatedEncoder后，输出特征图的通道数
}

# MobileNetV2
cfg_mnetv2 = {
    'name': 'mobilenetv2',
    'num_anchor': 6, # 5种锚
    'min_sizes': [8, 16, 32, 64, 128, 256], # 锚的边长：默认 5种锚 [32, 64, 128, 256, 512] [8, 16, 32, 64, 128, 256]
    'steps': 16, # 特征图缩放倍数（相对于原图）：16 （计算方式=锚的步长）
    'P_Thres': 0.3, # 正样本 筛选阈值：滤除 IoU＜P_Thres的 正样本; default 0.15
    'N_Thres': 0.3, # 负样本 筛选阈值：滤除 IoU＞N_Thres的 负样本; default 0.35
    'k_num': 5, # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 10，总数为 20
    'NegPos_ratio': 10, # 负正锚比例：调整为 10倍（避免负锚过多，影响训练）
    'variance': [0.1, 0.2], # 用于计算 真实边框回归值&真实面部标记点回归值，有利于 计算损失
    'clip': False,
    'cls_weight': 1.0, # 分类损失       所占的比重
    'loc_weight': 2.0, # 边框回归损失    所占的比重
    'landm_weight': 1.0, # 标记点回归损失  所占的比重
    'train_image_size': 640, # 缩减为 640
    'in_channel': 96, # BackBone输出特征图の通道数 = DilatedEncoder输入特征图の通道数
    'out_channel': 96 # 经过DilatedEncoder后，输出特征图的通道数
}

# MobileNetV3-Large
cfg_mnetv3_large = {
    'name': 'mobilenetv3large',
    'num_anchor': 6, # 5种锚
    'min_sizes': [8, 16, 32, 64, 128, 256], # 锚的边长：默认 5种锚 [32, 64, 128, 256, 512] [8, 16, 32, 64, 128, 256]
    'steps': 16, # 特征图缩放倍数（相对于原图）：16 （计算方式=锚的步长）
    'P_Thres': 0.3, # 正样本 筛选阈值：滤除 IoU＜P_Thres的 正样本; default 0.15
    'N_Thres': 0.3, # 负样本 筛选阈值：滤除 IoU＞N_Thres的 负样本; default 0.35
    'k_num': 5, # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 10，总数为 20
    'NegPos_ratio': 10, # 负正锚比例：调整为 10倍（避免负锚过多，影响训练）
    'variance': [0.1, 0.2], # 用于计算 真实边框回归值&真实面部标记点回归值，有利于 计算损失
    'clip': False,
    'cls_weight': 1.0, # 分类损失       所占的比重
    'loc_weight': 2.0, # 边框回归损失    所占的比重
    'landm_weight': 1.0, # 标记点回归损失  所占的比重
    'train_image_size': 640, # 缩减为 640
    'in_channel': 112, # BackBone输出特征图の通道数 = DilatedEncoder输入特征图の通道数
    'out_channel': 112 # 经过DilatedEncoder后，输出特征图的通道数
}

# MobileNetV3-Small
cfg_mnetv3_small = {
    'name': 'mobilenetv3small',
    'num_anchor': 6, # 5种锚
    'min_sizes': [8, 16, 32, 64, 128, 256], # 锚的边长：默认 5种锚 [32, 64, 128, 256, 512] [8, 16, 32, 64, 128, 256]
    'steps': 16, # 特征图缩放倍数（相对于原图）：16 （计算方式=锚的步长）
    'P_Thres': 0.3, # 正样本 筛选阈值：滤除 IoU＜P_Thres的 正样本; default 0.15
    'N_Thres': 0.3, # 负样本 筛选阈值：滤除 IoU＞N_Thres的 负样本; default 0.35
    'k_num': 5, # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 10，总数为 20
    'NegPos_ratio': 10, # 负正锚比例：调整为 10倍（避免负锚过多，影响训练）
    'variance': [0.1, 0.2], # 用于计算 真实边框回归值&真实面部标记点回归值，有利于 计算损失
    'clip': False,
    'cls_weight': 1.0, # 分类损失       所占的比重
    'loc_weight': 2.0, # 边框回归损失    所占的比重
    'landm_weight': 1.0, # 标记点回归损失  所占的比重
    'train_image_size': 640, # 缩减为 640
    'in_channel': 48, # BackBone输出特征图の通道数 = DilatedEncoder输入特征图の通道数
    'out_channel': 48 # 经过DilatedEncoder后，输出特征图的通道数
}

# ResNet18
cfg_resnet18 = {
    'name': 'resnet18',
    'num_anchor': 6, # 5种锚
    'min_sizes': [8, 16, 32, 64, 128, 256], # 锚的边长：默认 5种锚 [32, 64, 128, 256, 512]
    'steps': 16, # 特征图缩放倍数（相对于原图）：16 （计算方式=锚的步长）
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, # 输出特征层
    'P_Thres': 0.3, # 正样本 筛选阈值：滤除 IoU＜P_Thres的 正样本
    'N_Thres': 0.3, # 负样本 筛选阈值：滤除 IoU＞N_Thres的 负样本
    'k_num': 5, # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 4，总数为 8
    'NegPos_ratio': 10, # 负正锚比例：调整为 10倍（避免负锚过多，影响训练）
    'variance': [0.1, 0.2], # 用于计算 真实边框回归值&真实面部标记点回归值，有利于 计算损失
    'clip': False,
    'cls_weight': 1.0, # 分类损失       所占的比重
    'loc_weight': 2.0, # 人脸边框回归损失 所占的比重
    'landm_weight': 1.0, # 标记点回归损失  所占的比重
    'train_image_size': 640, # 训练图片边长，可选 640/960/1280
    'in_channel': 128, # BackBone输出特征图の通道数 = DilatedEncoder输入特征图の通道数
    'out_channel': 128 # 经过DilatedEncoder后，输出特征图的通道数
}

# History: not use ------------------------------------------------------

















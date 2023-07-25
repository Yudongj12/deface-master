
# 模型训练 相关函数
# 添加 改进的 均衡匹配机制

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_bbox import decode


# 定义函数：计算 锚的 左上点&右下点坐标<归一化>
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)

#------------------------------#
#   获得框的中心和宽高
#------------------------------#
def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)

#----------------------------------#
#   计算所有真实框和先验框的交面积
#----------------------------------#
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    #------------------------------#
    #   获得交矩形的左上角
    #------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #------------------------------#
    #   获得交矩形的右下角
    #------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    #-------------------------------------#
    #   计算先验框和所有真实框的重合面积
    #-------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


# 定义函数：计算 锚 & 真实边框的 IoU（交并比）
def jaccard(box_a, box_b):

    # 两个矩形框相交的 面积
    inter = intersect(box_a, box_b)

    # 计算 锚 & 真实边框 各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    return inter / union # 交并比


# 定义函数：编码操作（仅对正锚有意义），每个锚 相对于 真实边框的 回归值（中心点坐标 + 宽&高）（对应模型的 人脸边框回归结果，可用于计算损失）
# 输入声明：matched 每个锚 对应的 真实边框（左上点 + 右下点）；priors 每个锚（中心点 + 宽&高）
def encode(matched, priors, variances):
    # 中心点 编码：（真实框中心点坐标 - 锚中心点坐标）÷ 锚边长 ÷ 0.1
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] # :2 前2个
    g_cxcy /= (variances[0] * priors[:, 2:])
    # 宽&高 编码：log（真实框宽&高 ÷ 锚边长）÷ 0.2
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# 定义函数：编码操作（仅对 正锚、且对应真实边框有面部标记点 有意义），每个锚 相对于 真实面部标记点的 回归值（5点坐标）
#         （对应模型的 面部标记点回归结果，可用于计算损失）
def encode_landm(matched, priors, variances):
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy


# 定义函数：用于计算 锚的相对损失
def log_sum_exp(x):
    # 最大值
    x_max = x.data.max()
    # 按列求和，再取对数，输出维度=num×num_priors
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# 改进的 均衡匹配机制
# 定义函数：计算 （1张图像，但是loc_t会添加每张图像）所有锚/预测框的分类值conf & 正样本索引P_indices & 正样本の边框回归值loc &
#              含标记点的正样本索引P_landm_indices & 含标记点的正样本の标记点回归值landm（对应模型输出，可用于计算 损失）&
#              真实目标数Num_RealObj
# 输入声明：P_Th 正样本阈值；N_Th 负样本阈值；k_num 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 4，总数为 8；
#         truths 所有的真实边框；priors 所有的锚；variances 计算权重；labels 真实分类标签；landms 真实面部标记点；
#         loc_data_single 预测框；idx 批大小
def match(P_Th, N_Th, k_num, truths, priors, variance, labels, landms, loc_data_single, conf_t, P_indices_t, loc_t, P_landm_indices_t, landm_t, Num_RealObj_t, idx):

    # 解码操作：修改 预测框的格式，进行数值匹配
    loc_data_Temp = decode(loc_data_single, priors, variance)

    # 调用 jaccard函数，计算 所有真实边框 & 所有锚的 IoU：2阶Tensor，行数=真实目标数，列数=锚数
    IoU_Anchor = jaccard(
        truths,
        point_form(priors)  # 调用 函数point_form：计算 锚的 左上点&右下点坐标<归一化>
    )

    # 调用 jaccard函数，计算 所有真实边框 & 所有预测框的 IoU：2阶Tensor，行数=真实目标数，列数=预测框数
    IoU_Predict = jaccard(
        truths,
        point_form(loc_data_Temp)  # 调用 函数point_form：计算 锚的 左上点&右下点坐标<归一化>
    )

    # 真实目标数
    Num_RealObject = IoU_Anchor.size(0)
    # 锚数/预测框数
    Num_Anchor = IoU_Anchor.size(1)

    # 真实目标 对应的 2k个 锚（预测框）索引：1个Tensor 对应 1个目标（仅作为 中间值，后续会被修改！！）
    Sum_P_indices = []

    # 选取 正样本：已筛除 IoU＜0.15
    # --------------------------------------------------
    # 依次遍历 每个真实目标，获取 真实目标 对应的 4个原始锚 + 4个预测框（values IoU值 + indices 锚/预测框索引）
    # 注意：已筛除 IoU＜0.15的 锚（预测框）
    for i in range(Num_RealObject):
        # 第i个真实目标的 前k个 原始锚（values IoU值 + indices 锚索引）
        Selected_Anchor = torch.topk(IoU_Anchor[i, :], k=k_num)

        # 剔除 ＜0.1的锚，如果均不满足条件，则只选择 第1个
        # Selected_Anchor_values = Selected_Anchor.values[Selected_Anchor.values > P_Th]
        Selected_Anchor_indices = Selected_Anchor.indices[Selected_Anchor.values > P_Th]
        if Selected_Anchor_indices.size(0) == 0:
            Selected_Anchor_indices = torch.LongTensor([Selected_Anchor.indices[0]]).cuda()
        # （测试：不再剔除，因为可能没有达到条件的锚！！）
        # Selected_Anchor_indices = Selected_Anchor.indices[Selected_Anchor.values > 0]

        # 将选中的锚（预测框） 对应的列 元素值=-1，避免被再次选中
        IoU_Anchor[:, Selected_Anchor_indices] = -1.0
        IoU_Predict[:, Selected_Anchor_indices] = -1.0

        # 第i个真实目标的 前k个 预测框（values IoU值 + indices 预测框索引）
        # 注意：索引不与原始锚重叠
        Selected_Predict = torch.topk(IoU_Predict[i, :], k=k_num)

        # 剔除 ＜0.15的预测框: 筛选阈值为 P_Th + 0.05
        Selected_Predict_values = Selected_Predict.values[Selected_Predict.values > (P_Th + 0.05)]
        Selected_Predict_indices = Selected_Predict.indices[Selected_Predict.values > (P_Th + 0.05)]

        # 将选中的锚（预测框） 对应的列 元素值=-1，避免被再次选中
        IoU_Anchor[:, Selected_Predict_indices] = -1.0
        IoU_Predict[:, Selected_Predict_indices] = -1.0

        # 组合（索引值）
        Selected_All_indices = torch.cat([Selected_Anchor_indices, Selected_Predict_indices], dim=0)

        Sum_P_indices.append(Selected_All_indices.clone().detach())
    # --------------------------------------------------

    # 选取 负样本：已筛除 IoU＞0.35
    # --------------------------------------------------
    # 拼接 IoU_Anchor&IoU_Predict（被选择的正样本 对应的列 元素值=-1）
    IoU_All = torch.cat((IoU_Anchor, IoU_Predict), 0)  # 拼接Tensor

    # 计算 每列的最大值（锚/预测框的索引 对应的 最大IoU）
    Max_N = torch.max(IoU_All, dim=0)

    # 剔除 ＞0.35的锚/预测框（同时去掉 正样本）
    # 格式：1阶Tensor，1行，列数=锚数，元素值=True 负样本
    Selected_N_indices = (Max_N.values != -1) & (Max_N.values < N_Th)
    # --------------------------------------------------

    # 计算 正样本索引（合并为 1阶Tensor）& 正样本索引 对应的 真实目标索引（1阶Tensor）
    # --------------------------------------------------
    # 正样本索引（合并为 1阶Tensor）
    P_indices = torch.cat(Sum_P_indices, 0)
    # P_indices = torch.stack(Sum_P_indices).reshape(1, -1).squeeze()

    for j in range(Num_RealObject):
        if Sum_P_indices[j] != []:
            Sum_P_indices[j][:] = j

    # 正样本索引 对应的 真实目标索引（1阶Tensor）
    P_True = torch.cat(Sum_P_indices, 0)
    # P_True = torch.stack(Sum_P_indices).reshape(1, -1).squeeze()
    # --------------------------------------------------

    # 含标记点的 正样本索引（合并为 1阶Tensor）
    P_landm_indices = P_indices[labels[P_True] == 1]
    # 含标记点的 正样本索引 对应的 真实目标索引（1阶Tensor）
    P_landm_True = P_True[labels[P_True] == 1]
    # 含标记点的正样本の标记点回归值landm
    if(P_landm_indices.numel() > 0):
        landm = encode_landm(landms[P_landm_True], priors[P_landm_indices], variance).cuda()
    else:
        landm = torch.Tensor().cuda()

    # 批中图像 叠加操作
    # P_landm_indices_t[idx] = P_landm_indices
    # landm_t[idx] = landm
    P_landm_indices_t.append(P_landm_indices)
    landm_t.append(landm)

    # 正样本 相对于 真实目标的 边框回归值：2阶Tensor，行数=正样本数，列数=4
    loc = encode(truths[P_True], priors[P_indices], variance)
    # 批中图像 叠加操作
    # P_indices_t[idx] = P_indices
    # loc_t[idx] = loc
    P_indices_t.append(P_indices)
    loc_t.append(loc)

    # 所有锚/预测框 对应的 标志：1阶Tensor，列数=锚数/预测框数；-1 正样本/前景，不含标记点；1 正样本/前景，含标记点；0 负样本/背景；-2 忽略样本
    # 修改：正样本均为 1
    conf = (torch.zeros(Num_Anchor) - 2).cuda()  # 初始化，元素值均为 -2
    # conf[P_indices] = labels[P_True]  # 添加 正样本标志
    conf[P_indices] = 1  # 添加 正样本标志
    conf[Selected_N_indices] = 0  # 添加 负样本标志
    # 批中图像 叠加操作
    conf_t.append(conf)

    # 真实目标数
    Num_RealObj = Num_RealObject
    # 批中图像 叠加操作
    Num_RealObj_t.append(Num_RealObj)


# 定义 MultiBoxLoss类：计算 多任务损失
class MultiBoxLoss(nn.Module):
    # 初始化：num_classes 类别数 2<人脸/背景>；P_Th 正样本阈值；N_Th 负样本阈值；
    #        k_num 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 4，总数为 8；
    #        neg_pos 正负锚比例=1:7；variance 计算权重；cuda GPU
    def __init__(self, num_classes, P_Th, N_Th, k_num, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()

        # 对retinaface而言，num_classes等于2
        self.num_classes = num_classes

        # 正样本 筛选阈值：滤除 IoU＜P_Th的 正样本
        self.P_Th = P_Th
        # 负样本 筛选阈值：滤除 IoU＞N_Th的 负样本
        self.N_Th = N_Th
        # 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 4，总数为 8
        self.k_num = k_num

        # 正负样本比例 = 1:25（避免负锚过多，影响训练）
        self.negpos_ratio = neg_pos
        # 计算权重
        self.variance = variance
        # 是否使用 GPU（默认 使用）
        self.cuda = cuda

    # 输入声明：predictions 模型输出结果；priors 所有锚的位置信息（中心点坐标 + 宽&高）<归一化>；targets 人脸标注信息
    def forward(self, predictions, priors, targets):

        # 模型输出结果：人脸边框回归结果，2分类结果，面部标记点回归结果
        loc_data, conf_data, landm_data = predictions

        # num 批大小（1个批包含的 图像数）
        num = loc_data.size(0)

        # 初始化 所有锚/预测框的分类值（批中所有图像）：-1 正样本/前景，不含标记点；1 正样本/前景，含标记点；0 负样本/背景；-2 忽略样本
        conf_t = []
        # 初始化 正样本索引（批中所有图像）
        P_indices_t = []
        # 初始化 正样本の边框回归值（批中所有图像）
        loc_t = []
        # 初始化 含标记点的正样本索引（批中所有图像）
        P_landm_indices_t = []
        # 初始化 含标记点的正样本の标记点回归值（批中所有图像）
        landm_t = []
        # 初始化 真实目标数（批中所有图像）
        Num_RealObj_t = []

        # 遍历 每张图像，获取 真实边框truths + 真实分类标签labels + 真实面部标记点landms
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data

            # 单张图像的 人脸边框回归预测结果：只做简单的数据复制，既不数据共享，也不对梯度共享
            loc_data_single = loc_data[idx].clone().detach()

            # 所有的锚：只做简单的数据复制，既不数据共享，也不对梯度共享
            defaults = priors.clone().detach()


            # 调用 函数match，计算 所有锚/预测框的分类值conf & 正样本索引P_indices & 正样本の边框回归值loc &
            #               含标记点的正样本索引P_landm_indices & 含标记点的正样本の标记点回归值landm & 真实目标数Num_RealObj
            # 输入声明：P_Th 正样本阈值；N_Th 负样本阈值；k_num 每个真实目标对应的 正样本数量（锚 or 预测框）：暂定 4，总数为 8；
            #         truths 所有的真实边框；defaults 所有的锚；variances 计算权重；
            #         labels 真实分类标签；landms 真实面部标记点；loc_data_single 预测框；idx 批大小
            match(self.P_Th, self.N_Th, self.k_num, truths, defaults, self.variance, labels, landms, loc_data_single, conf_t, P_indices_t, loc_t, P_landm_indices_t, landm_t, Num_RealObj_t, idx)

        # 计算 面部标记点回归损失
        # ————————————————————————————————————————————————
        # 模型输出面部标记点回归值
        landm_p = torch.Tensor([]).cuda()
        for idx1 in range(num):
            landm_p = torch.cat((landm_p, landm_data[idx1, P_landm_indices_t[idx1], :]), 0)

        # 真实面部标记点回归值: Only use 2 eye + 1 nose
        landm_t_cuda = torch.cat(landm_t, 0).cuda()
        landm_t_eye_cuda = landm_t_cuda[:, :6] # only select eye + nose

        # Smooth L1损失
        loss_landm = F.smooth_l1_loss(landm_p, landm_t_eye_cuda, reduction='sum')
        # ————————————————————————————————————————————————

        # 计算 人脸边框回归损失
        # ————————————————————————————————————————————————
        # 模型输出人脸边框回归值（中心点坐标 + 宽&高）
        loc_p = torch.Tensor([]).cuda()
        for idx2 in range(num):
            loc_p = torch.cat((loc_p, loc_data[idx2, P_indices_t[idx2], :]), 0)

        # 真实人脸边框回归值（中心点坐标 + 宽&高）
        loc_t_cuda = torch.cat(loc_t, 0).cuda()

        # Smooth L1损失
        loss_l = F.smooth_l1_loss(loc_p, loc_t_cuda, reduction='sum')
        # ————————————————————————————————————————————————

        # 计算 分类损失（仅该损失需要考虑 正负锚比例！！）
        # ————————————————————————————————————————————————
        # 批中所有图像的 正样本&负样本 对应的 真实标签
        My_PN_Real_All = []

        # 批中所有图像的 正样本&负样本 对应的 预测标签
        My_PN_Predict_All = []

        # 分别提取 批中每张图像的 正样本&负样本
        for kk in range(num):
            # 加入 正样本
            My_P_Real = conf_t[kk][conf_t[kk] == 1]  # 正样本 对应的 真实标签：1阶Tensor，元素数=正样本数，元素值=1
            My_P_Predict = conf_data[kk, conf_t[kk] == 1, :]  # 正样本 对应的 预测标签：2阶Tensor，行数=正样本数，列数=2

            Num_Positive_test = My_P_Real.numel() # 正样本数（单张图像）

            # 加入 负样本
            Unsure_N_Real = conf_t[kk][conf_t[kk] == 0]  # 负样本 对应的 真实标签：1阶Tensor，元素数=负样本数，元素值=0
            Unsure_N_Real = Unsure_N_Real.type(torch.int64)
            Unsure_N_Predict = conf_data[kk, conf_t[kk] == 0, :]  # 负样本 对应的 预测标签：2阶Tensor，行数=负样本数，列数=2

            # 计算 每个负样本 对应于真实标签的 损失：相对参考值 - 每个负样本相对于真实标签的预测值
            loss_c = log_sum_exp(Unsure_N_Predict) - Unsure_N_Predict.gather(1, Unsure_N_Real.view(-1, 1))  # 1阶Tensor，元素数=负样本数
            loss_c = loss_c.squeeze()

            # 按损失进行 递减排序，loss_idx 排序后的 负样本索引（相对）
            _, loss_idx = loss_c.sort(-1, descending=True)
            # 对负样本索引（相对）进行 递增排序：负样本索引位置的元素值=该负样本的排名
            _, idx_rank = loss_idx.sort(-1)

            # 确定 当前图像中 所需负样本的数量
            # Choice1: 真实目标数 × 10 × 10（限制 负样本数量，有利于训练）
            # Num_N = Num_RealObj_t[kk] * 10 * 10
            # Choice2: 正样本数 × negpos_ratio（限制 负样本数量，有利于训练）
            Num_N = Num_Positive_test * self.negpos_ratio

            # 筛选 负样本：1阶Tensor，元素数=负样本数（未筛选），元素值=True 被选中，元素值=False 落选
            neg = idx_rank < Num_N

            My_N_Real = Unsure_N_Real[neg]  # 负样本 对应的 真实标签：1阶Tensor，元素数=负样本数（已筛选），元素值=0
            My_N_Predict = Unsure_N_Predict[neg, :]  # 负样本 对应的 预测标签：2阶Tensor，行数=负样本数（已筛选），列数=2

            My_PN_Real_All.append(torch.cat((My_P_Real, My_N_Real)))
            My_PN_Predict_All.append(torch.cat((My_P_Predict, My_N_Predict)))

        # 列表 → Tensor
        My_PN_Real_cuda = torch.cat(My_PN_Real_All, 0).cuda()
        My_PN_Real_cuda = My_PN_Real_cuda.type(torch.int64)
        My_PN_Predict_cuda = torch.cat(My_PN_Predict_All, 0).cuda()

        # 计算 交叉熵损失
        loss_cls = F.cross_entropy(My_PN_Predict_cuda, My_PN_Real_cuda, reduction='sum')
        # ————————————————————————————————————————————————

        # 平均损失值：对于 loss_l & loss_c，÷ 正样本数；对于 loss_landm，÷ 含标记点的 正样本数
        # 批中所有图像 正样本数总和
        N = torch.cat(P_indices_t, 0).numel()
        loss_l /= N
        loss_cls /= N

        # 批中所有图像 含标记点的 正样本数总和
        N1 = torch.cat(P_landm_indices_t, 0).numel()
        loss_landm /= N1

        # 输出声明：loss_l 人脸边框回归损失；loss_c 分类损失；loss_landm 面部标记点回归损失
        return loss_l, loss_cls, loss_landm


# 定义 函数：初始化 模型权重
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# 定义 函数：学习率 调整方式
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3) # default 3; 可修改 Warm-Up 迭代次数
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


# 定义 函数：优化器 设置
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

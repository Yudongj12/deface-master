
# 定义函数fit_one_epoch：进行1次迭代训练（计算损失 + 反向传播 + 保存模型文件）

import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, criterion, epoch, epoch_step, gen, Epoch, anchors, cfg, cuda, save_period, save_dir):
    total_r_loss = 0
    total_c_loss = 0
    total_landmark_loss = 0

    print('Start Train')
    # 显示 实时训练进度 & 相关信息
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:

        # 以 batch批 遍历 训练集的所有样本，训练模型
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: # epoch_step 训练集的 总批数
                break
            images, targets = batch[0], batch[1]
            if len(images) == 0:
                continue
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # 清零梯度
            optimizer.zero_grad()
            # 前向计算，获取 预测结果
            out = model_train(images)

            # 计算 多任务损失：2×人脸边框回归损失 + 二分类损失 + 面部标记点回归损失
            # 调用 MultiBoxLoss函数 实现（criterion实例）
            # 初始化：设置 类别数=2<人脸/背景>、正负锚判断阈值、正负锚比例、计算权重、使用GPU？
            # 输入声明：模型输出结果；所有锚的位置信息（中心点坐标 + 宽&高）<归一化>；人脸标注信息
            # 输出声明：r_loss 人脸边框回归损失；c_loss 分类损失；loss_landm 面部标记点回归损失
            r_loss, c_loss, landm_loss = criterion(out, anchors, targets)

            # 总损失
            loss = cfg['cls_weight'] * c_loss + cfg['loc_weight'] * r_loss + cfg['landm_weight'] * landm_loss
            # 反向传播
            loss.backward()
            # 获取 梯度，更新 模型参数
            optimizer.step()

            # 分别计算 各任务总损失，打印显示
            total_c_loss += cfg['cls_weight'] * c_loss.item()
            total_r_loss += cfg['loc_weight'] * r_loss.item()
            total_landmark_loss += cfg['landm_weight'] * landm_loss.item()
            
            pbar.set_postfix(**{'Conf Loss'         : total_c_loss / (iteration + 1), 
                                'Regression Loss'   : total_r_loss / (iteration + 1), 
                                'LandMark Loss'     : total_landmark_loss / (iteration + 1), 
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)

    # 个人修改：不再记录并保存损失
    loss_history = 0
    # loss_history.append_loss(epoch + 1, (total_c_loss + total_r_loss + total_landmark_loss) / epoch_step)
    print('Saving state, iter:', str(epoch + 1))

    # 每隔 save_period次迭代，保存模型文件
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'Epoch%d-Total_Loss%.4f.pth'%((epoch + 1), (total_c_loss + total_r_loss + total_landmark_loss) / epoch_step)))

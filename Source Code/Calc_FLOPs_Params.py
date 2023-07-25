
import torch
import tqdm
import torch.nn as nn
from nets.retinaface import RetinaFace
from utils.config import cfg_mnet, cfg_re50, cfg_mnetv3_large, cfg_mnetv3_small

from thop import profile # FLOPs & Params


# Calculate FLOPs & Params
def FLOPS_and_Params(model, min_size, max_size, device):
    x = torch.randn(1, 3, min_size, max_size).to(device)
    print('-----------------------------')

    flops, params = profile(model, inputs=(x, ))
    print('FLOPs: {:.2f} B'.format(flops / 1e9))
    print('Params: {:.2f} M'.format(params / 1e6))

    print('-----------------------------')


if __name__ == '__main__':

    # Calculate FLOPs & Params (Default Use CUDA)
    Test_retinaface = RetinaFace(cfg=cfg_mnetv3_small)

    Test_retinaface.load_state_dict(torch.load('logs/V3Small_6.7_Bad/Epoch150-Total_Loss23.4635.pth', map_location=torch.device("cuda")))

    Test_retinaface = Test_retinaface.cuda()

    FLOPS_and_Params(model=Test_retinaface, min_size=1280, max_size=1280, device=torch.device("cuda"))

















from networks.unet import UNet
import torch.nn as nn
from networks.unet_mfa import UNet_MFA

def net_factory(net_type="unet", in_chns=1, class_num=3, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    return net

def MFANet(in_chns=1, class_num=2, ema=False):
    net = UNet_MFA(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net
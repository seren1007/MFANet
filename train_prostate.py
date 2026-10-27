import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio
from dataloaders.transform import FullAugmentor
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from utils.MFA_utils import boundary_laplacian_loss, boundary_LoG_loss, sup_loss
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import MFANet, net_factory
from utils import losses, ramps, val_2d

parser = argparse.ArgumentParser()


parser.add_argument('--root_path', type=str, default='./dataset/ProstateX/',help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MFANet', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--u_weight', type=float, default=1.0, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=3)
boundary_loss = boundary_laplacian_loss()
boundary_loss_u = boundary_LoG_loss()
ce_loss = CrossEntropyLoss()


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_LargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_LargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "12": 196, "16": 251, "24": 401, "32": 483}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    model = MFANet(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            out,out_stages_0,out_stages_1,out_stages_2,out_stages_3 = model(volume_batch)
            outputs_soft = torch.softmax(out, dim=1)

            loss_dice = dice_loss(out, label_batch.unsqueeze(1))
            loss_ce = ce_loss(outputs_soft,label_batch[:].long())

            boundery_bce_loss = 0.0
            boundery_dice_loss = 0.0

            boundery_bce_loss_0, boundery_dice_loss_0 = boundary_loss(out_stages_0, label_batch)
            boundery_bce_loss += boundery_bce_loss_0
            boundery_dice_loss += boundery_dice_loss_0

            boundery_bce_loss_1, boundery_dice_loss_1 = boundary_loss(out_stages_1, label_batch)
            boundery_bce_loss += boundery_bce_loss_1
            boundery_dice_loss += boundery_dice_loss_1

            boundery_bce_loss_2, boundery_dice_loss_2 = boundary_loss(out_stages_2, label_batch)
            boundery_bce_loss += boundery_bce_loss_2
            boundery_dice_loss += boundery_dice_loss_2

            boundery_bce_loss_3, boundery_dice_loss_3 = boundary_loss(out_stages_3, label_batch)
            boundery_bce_loss += boundery_bce_loss_3
            boundery_dice_loss += boundery_dice_loss_3

            loss = (loss_dice + loss_ce) / 2 + boundery_dice_loss + boundery_bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f, detail_loss: %f' % (
            iter_num, loss, loss_dice, loss_ce, boundery_dice_loss + boundery_bce_loss))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))

    model = MFANet(in_chns=1, class_num=num_classes)
    ema_model = MFANet(in_chns=1, class_num=num_classes, ema=True)

    strong_perturbation = FullAugmentor()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img = volume_batch[:args.labeled_bs]
            lab = label_batch[:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]
            # strongly perturbation
            unimg_aug = strong_perturbation.forward_image(unimg)
            ema_inputs = unimg
            model_inputs = torch.cat([img, unimg_aug], dim=0)
            with torch.no_grad():
                ema_output,_,_,_,_= ema_model(ema_inputs)
                ema_output_mask = get_masks(ema_output, nms=1)

            outputs,out_encoder_0, out_encoder_1, out_encoder_2, out_encoder_3 = model(model_inputs)
            outputs_l, out_encoder_0_l,out_encoder_1_l, out_encoder_2_l, out_encoder_3_l = outputs[:args.labeled_bs], out_encoder_0[:args.labeled_bs],\
                out_encoder_1[:args.labeled_bs], out_encoder_2[:args.labeled_bs], out_encoder_3[:args.labeled_bs]
            outputs_u,out_encoder_0_u, out_encoder_1_u, out_encoder_2_u, out_encoder_3_u = outputs[args.labeled_bs:],out_encoder_0[args.labeled_bs:],\
                out_encoder_1[args.labeled_bs:],out_encoder_2[args.labeled_bs:], out_encoder_3[args.labeled_bs:]

            loss_l_dice_ce = sup_loss(outputs_l, lab)

            loss_u_ce = ce_loss(outputs_u, ema_output_mask.long())
            loss_u_cos = (1.0 - F.cosine_similarity(outputs_u, ema_output, dim=1)).mean()

            #
            boundery_bce_loss_l = 0.0
            boundery_dice_loss_l = 0.0
            boundery_bce_loss_0_l, boundery_dice_loss_0_l = boundary_loss(out_encoder_0_l, lab)
            boundery_bce_loss_l += boundery_bce_loss_0_l
            boundery_dice_loss_l += boundery_dice_loss_0_l

            boundery_bce_loss_1_l, boundery_dice_loss_1_l = boundary_loss(out_encoder_1_l, lab)
            boundery_bce_loss_l += boundery_bce_loss_1_l
            boundery_dice_loss_l += boundery_dice_loss_1_l

            boundery_bce_loss_2_l, boundery_dice_loss_2_l = boundary_loss(out_encoder_2_l, lab)
            boundery_bce_loss_l += boundery_bce_loss_2_l
            boundery_dice_loss_l += boundery_dice_loss_2_l

            boundery_bce_loss_3_l, boundery_dice_loss_3_l = boundary_loss(out_encoder_3_l, lab)
            boundery_bce_loss_l += boundery_bce_loss_3_l
            boundery_dice_loss_l += boundery_dice_loss_3_l

            boundery_bce_loss_u = 0.0
            boundery_dice_loss_u = 0.0

            boundery_bce_loss_0_u, boundery_dice_loss_0_u = boundary_loss_u(out_encoder_0_u,ema_output_mask)
            boundery_bce_loss_u += boundery_bce_loss_0_u
            boundery_dice_loss_u += boundery_dice_loss_0_u

            boundery_bce_loss_1_u, boundery_dice_loss_1_u = boundary_loss_u(out_encoder_1_u, ema_output_mask)
            boundery_bce_loss_u += boundery_bce_loss_1_u
            boundery_dice_loss_u += boundery_dice_loss_1_u

            boundery_bce_loss_2_u, boundery_dice_loss_2_u = boundary_loss_u(out_encoder_2_u, ema_output_mask)
            boundery_bce_loss_u += boundery_bce_loss_2_u
            boundery_dice_loss_u += boundery_dice_loss_2_u

            boundery_bce_loss_3_u, boundery_dice_loss_3_u = boundary_loss_u(out_encoder_3_u, ema_output_mask)
            boundery_bce_loss_u += boundery_bce_loss_3_u
            boundery_dice_loss_u += boundery_dice_loss_3_u

            loss_l = loss_l_dice_ce
            loss_u = 0.5*loss_u_ce + 0.5*loss_u_cos
            loss_b = boundery_dice_loss_l + boundery_bce_loss_l + boundery_dice_loss_u + boundery_bce_loss_u

            loss = loss_l + loss_u + loss_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_l, iter_num)
            writer.add_scalar('info/mix_ce', loss_u, iter_num)

            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f,loss_u_pcs: %03f'
                         % (iter_num, loss, loss_l_dice_ce, loss_u_ce,loss_u_cos))

            if iter_num % 20 == 0:
                image = volume_batch[args.labeled_bs:][1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[args.labeled_bs:], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = volume_batch[args.labeled_bs:][1, ...].unsqueeze(0) * 50


                image_l = volume_batch[:args.labeled_bs][1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(outputs[:args.labeled_bs].float(), dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = outputs_l[1, ...].unsqueeze(0) * 50


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/ProstateX_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/ProstateX_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('train_prostate.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)





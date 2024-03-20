"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""
from typing import List

from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import datetime

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation, DGSegmentation, iSSegmentation, VhSegmentation, PdSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.utils import AverageMeter, ExponentialMovingAverage
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils import KDLoss, WBCELoss, ACLoss, Dynamic_Loss

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy

torch.backends.cudnn.benchmark = True


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DeepGlobe2018',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='deepglobe',
                        choices=['voc', 'ade', 'deepglobe', 'iSAID', 'vaihingen', 'potsdam'],
                        help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50,
                        help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--test_batch_size", type=int, default=1,
                        help='batch size for test (default: 1)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=30,
                        help="epoch interval for eval (default: 30)")
    parser.add_argument("--val_times", type=int, default=30,
                        help="epoch interval for eval (default: 30)")

    # CIL options
    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", action='store_true', help="enable network freezing")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--w_transfer", action='store_true', help="enable weight transfer")
    parser.add_argument("--unknown", action='store_true', help="enable unknown modeling")
    parser.add_argument("--meta", action='store_true', help="enable meta block")
    parser.add_argument("--train_branch", type=str, default='both', choices=['ce', 'bce', 'both'], help="train branch")
    parser.add_argument("--infer_branch", type=str, default='both', choices=['ce', 'bce', 'both'], help="infer branch")
    parser.add_argument("--kd", type=float, default=2)
    parser.add_argument("--dkd", type=float, default=2)
    parser.add_argument("--test_on_val", action='store_true', help="whether test on val set")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="set ema model decay rate")
    parser.add_argument("--ema_thresh", type=float, default=0.5,
                        help="ema_model confidence threshold for pseudo-labeling")
    parser.add_argument("--ema_loss", type=float, default=1,
                        help="ema_model confidence threshold for pseudo-labeling")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    test_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    elif opts.dataset == 'deepglobe':
        dataset = DGSegmentation
    elif opts.dataset == 'iSAID':
        dataset = iSSegmentation
    elif opts.dataset == 'vaihingen':
        dataset = VhSegmentation
    elif opts.dataset == 'potsdam':
        dataset = PdSegmentation
    else:
        raise NotImplementedError

    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)

    dataset_dict['val'] = dataset(opts=opts, image_set='val', transform=val_transform, cil_step=opts.curr_step)

    dataset_dict['test'] = dataset(opts=opts, image_set='test_val' if opts.test_on_val else 'test',
                                   transform=test_transform, cil_step=opts.curr_step)

    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform,
                                         cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(loader):

            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs1, outputs2, outputs2_pos, outputs2_neg = model(images)

            # logit = torch.sigmoid(outputs2)
            # pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
            # idx = (logit[:, 1:] > 0.3).float()  # logit: [N, C, H, W]
            # idx = idx.sum(dim=1)  # logit: [N, H, W]
            # pred[idx == 0] = 0  # set background (non-target class)
            # pred = pred.cpu().numpy()

            if opts.loss_type == 'bce_loss':
                outputs1 = torch.sigmoid(outputs1)
                outputs2 = torch.sigmoid(outputs2)
            else:
                outputs1 = torch.softmax(outputs1, dim=1)
                outputs2 = torch.softmax(outputs2, dim=1)

            # remove unknown label
            if opts.unknown:
                outputs1[:, 1] += outputs1[:, 0]
                outputs1 = outputs1[:, 1:]
                outputs2[:, 1] += outputs2[:, 0]
                outputs2 = outputs2[:, 1:]
            # sigmoid加和不好
            # 1背景 0 unknown

            if opts.infer_branch == 'both':
                outputs = outputs1 + outputs2
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            elif opts.infer_branch == 'bce':

                logit = torch.sigmoid(outputs2)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)
                preds = pred.cpu().numpy()

            else:
                outputs = outputs1
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False

    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes: List[int] = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step + 1)]
    if opts.unknown:  # re-labeling: [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0] - 1] + opts.num_classes[1:]
    else:  # [unknown, ...]
        opts.num_classes = [1, opts.num_classes[0] - 1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0

    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)),
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step + 1))
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print("  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,
                                  bn_freeze=bn_freeze, with_meta=opts.meta)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.curr_step > 0:
        """ load previous model """
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride,
                                           bn_freeze=bn_freeze, with_meta=opts.meta)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None

    # Set up metrics
    #                            到当前的学到的classes为止
    metrics = StreamSegMetrics(sum(opts.num_classes) - 1 if opts.unknown else sum(opts.num_classes),
                               dataset=opts.dataset)
    # sum((15,1,1,...))
    print(model.classifier.head)
    # model是组合好的

    # Set up optimizer & parameters
    if opts.freeze and opts.curr_step > 0:
        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.backbone.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

        if opts.meta:
            for param in model.backbone.meta_layer[:-opts.num_classes[-1]].parameters():
                param.requires_grad = False

        training_params = [{'params': model.classifier.parameters(), 'lr': opts.lr}]
        training_params.append({'params': model.backbone.parameters(), 'lr': opts.lr})

    else:
        training_params = [{'params': model.classifier.parameters(), 'lr': 0.01}]
        training_params.append({'params': model.backbone.parameters(), 'lr': opts.lr})

    # if opts.freeze and opts.curr_step > 0:
    #     for param in model_prev.parameters():
    #         param.requires_grad = False
    #
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     for param in model.classifier.head[-1].parameters(): # classifier for new class
    #         param.requires_grad = True  # classifier.head is module container:?
    #
    #     training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': opts.lr}]
    #
    #     if opts.unknown:
    #         for param in model.classifier.head[0].parameters(): # unknown
    #             param.requires_grad = True
    #         training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
    #
    #         for param in model.classifier.head[1].parameters(): # background
    #             param.requires_grad = True
    #         training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})
    #
    # else:
    #     training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
    #                        {'params': model.classifier.parameters(), 'lr': 0.01}]

    optimizer = torch.optim.SGD(params=training_params,
                                lr=opts.lr,
                                momentum=0.9,
                                weight_decay=opts.weight_decay,
                                nesterov=True)

    print("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("-----------------------------------------------")

    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0

    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"

    if opts.curr_step > 0:  # previous step checkpoint
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step - 1)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)

        # if opts.unknown and opts.w_transfer:
        if opts.w_transfer:
            # weight transfer : from unknown to new-class
            print("... weight transfer")
            curr_head_num = len(model.classifier.head) - 1

            checkpoint[f"classifier.head.{curr_head_num}.0.weight"] = checkpoint["classifier.head.0.0.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.weight"] = checkpoint["classifier.head.0.1.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.bias"] = checkpoint["classifier.head.0.1.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_mean"] = checkpoint[
                "classifier.head.0.1.running_mean"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_var"] = checkpoint["classifier.head.0.1.running_var"]

            last_conv_weight = model.state_dict()[f"classifier.head.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.head.{curr_head_num}.3.bias"]

            for i in range(opts.num_classes[-1]):
                last_conv_weight[i] = checkpoint["classifier.head.0.3.weight"]
                last_conv_bias[i] = checkpoint["classifier.head.0.3.bias"]

            checkpoint[f"classifier.head.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.head.{curr_head_num}.3.bias"] = last_conv_bias

            # head2
            last_conv_weight2 = model.state_dict()[f"classifier.head2.{curr_head_num}.weight"]
            last_conv_bias2 = model.state_dict()[f"classifier.head2.{curr_head_num}.bias"]

            for i in range(opts.num_classes[-1]):
                last_conv_weight2[i] = checkpoint["classifier.head2.0.weight"]
                last_conv_bias2[i] = checkpoint["classifier.head2.0.bias"]

            checkpoint[f"classifier.head2.{curr_head_num}.weight"] = last_conv_weight2
            checkpoint[f"classifier.head2.{curr_head_num}.bias"] = last_conv_bias2

        model.load_state_dict(checkpoint, strict=False)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")

    model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()
        model_ema = ExponentialMovingAverage(model, opts.ema_decay)
        model_ema = model_ema.to(device)
        model_ema.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)

        # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1

    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size if opts.test_on_val else opts.test_batch_size,
        shuffle=False, num_workers=4)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))

    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)

    total_itrs = opts.train_epoch * len(train_loader)
    print(f"... train epoch : {opts.train_epoch} , iterations : {total_itrs} , val_interval : {opts.val_interval}",
          f' , print_interval : {opts.print_interval} , val_times : {opts.val_times}')

    # ==========   Train Loop   ==========#
    if opts.test_only:
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader,
                              device=device, metrics=metrics)

        print(metrics.to_str(test_score))
        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())

        first_cls = len(get_tasks(opts.dataset, opts.task, 0))  # 15-1 task -> first_cls=16
        print(f"...from 0 to {first_cls - 1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(
            f"...from {first_cls} to {len(class_iou) - 1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls - 1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(
            f"...from {first_cls} to {len(class_iou) - 1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
        return

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'warm_poly':
        warmup_iters = int(total_itrs * 0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, reduction='mean')

    # if opts.unknown:
    #     pos_weight = torch.tensor([1, 1, 10, 8, 15, 11, 12, 13], device=device)
    # else:
    #     pos_weight = torch.tensor([1, 10, 8, 15, 11, 12, 13], device=device)
    pos_weight = 30 * torch.ones(sum(opts.num_classes), device=device)
    kdloss = KDLoss()
    bceloss = WBCELoss(pos_weight=pos_weight, n_old_classes=sum(opts.num_classes[:-1]),
                       n_new_classes=opts.num_classes[-1])
    acloss = ACLoss()
    dynamic_loss = Dynamic_Loss(opts.num_classes)

    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)

    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    avg_celoss = AverageMeter()
    avg_kdloss = AverageMeter()
    avg_bceloss = AverageMeter()
    avg_acloss = AverageMeter()
    avg_dkdloss_pos = AverageMeter()
    avg_dkdloss_neg = AverageMeter()
    avg_emaloss = AverageMeter()

    model.train()
    save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))

    def get_threshhold(cur_model, train_loader, ratio):
        train_iter = iter(train_loader)
        class_possibilities = [np.array([], dtype=np.float32) for _ in range(sum(opts.num_classes))]

        for images, labels, _, _ in train_iter:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=opts.amp):
                with torch.no_grad():
                    outputs1, _, _, _ = cur_model(images)
                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs1).detach()
                else:
                    pred_prob = torch.softmax(outputs1, 1).detach()

                pseudo_posibilities, pseudo_labels = pred_prob.max(dim=1)
                pseudo_labels = torch.where((labels >=sum(opts.num_classes[:-1])), labels, pseudo_labels)
                index = torch.where((labels == 255), 0, pseudo_labels)
                pseudo_posibilities = torch.gather(pred_prob, dim=1, index=index.unsqueeze(1)).squeeze(1)
                for label in torch.unique(pseudo_labels):
                    if label == 255:
                        continue
                    p = pseudo_posibilities[(pseudo_labels == label)].sort().values
                    slice = max(1,len(p)//100)
                    p = p[::slice].cpu().numpy()
                    class_possibilities[label] = np.concatenate((p, class_possibilities[label]))
#                 for i in range(sum(opts.num_classes)):
#                     p = pseudo_posibilities[pseudo_labels == i].cpu().numpy()
#                     class_possibilities[i] = np.concatenate((p, class_possibilities[i]))

        t = []
        for i in range(sum(opts.num_classes)):
            if len(class_possibilities[i]) == 0:
                print(f'there is no {i} class pseudo pixels')
                t.append(0)
            else:
                class_possibilities[i].sort()
                t.append(class_possibilities[i][-int(len(class_possibilities[i]) * ratio)])
        print('\n pseudo thresh is', t)
        return t

    thresh_ratio = [0.2, 0.4, 0.6, 0.8, 1]
    ratio_iter = iter(thresh_ratio)
    change_itrs = int(total_itrs / len(thresh_ratio)) + 1
    # =====  Train  =====
    while cur_itrs < total_itrs:

        if opts.curr_step > 0 and cur_itrs % change_itrs == 0:
            ratio = next(ratio_iter)
            pseudo_thresh = get_threshhold(model_ema, train_loader, ratio)


        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()

        """ data load """
        try:
            images, labels, sal_maps, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, sal_maps, _ = train_iter.next()
            cur_epochs += 1
            avg_loss.reset()
            avg_time.reset()
            avg_celoss.reset()
            avg_bceloss.reset()
            avg_acloss.reset()
            avg_kdloss.reset()
            avg_dkdloss_pos.reset()
            avg_dkdloss_neg.reset()
            avg_emaloss.reset()

        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)

        """ memory """
        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, m_sal_maps, _ = mem_iter.next()

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_sal_maps = m_sal_maps.to(device, dtype=torch.long, non_blocking=True)

            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()  # 0 -- batch_size-1 随机替换一半
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]
            sal_maps[rand_index, ...] = m_sal_maps[rand_index, ...]

        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=opts.amp):

            # outputs = model(images, pseudo_labels)

            loss_ce = torch.tensor(0, device=device, dtype=torch.float)
            loss_bce = torch.tensor(0, device=device, dtype=torch.float)
            loss_ac = torch.tensor(0, device=device, dtype=torch.float)
            loss_kd = torch.tensor(0, device=device, dtype=torch.float)
            loss_dkd_pos = torch.tensor(0, device=device, dtype=torch.float)
            loss_dkd_neg = torch.tensor(0, device=device, dtype=torch.float)
            loss_ema = torch.tensor(0, device=device, dtype=torch.float)

            if opts.pseudo and opts.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs1_prev, outputs2_prev, outputs2_pos_prev, outputs2_neg_prev = model_prev(images)
                    outputs1_ema, _, _, _ = model_ema(images)

                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs1_prev).detach()
                    ema_prob = torch.sigmoid(outputs1_ema).detach()
                else:
                    pred_prob = torch.softmax(outputs1_prev, 1).detach()
                    ema_prob = torch.softmax(outputs1_ema, 1).detach()

                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where(
                    (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh),
                    pred_labels,
                    labels)

                # ema_labels = torch.where(((labels == ema_labels) & (labels > fg_idx)) | (
                #             (labels <= fg_idx) & (ema_scores >= opts.ema_thresh))
                #                          , ema_labels, 255)
                # ema_scores, ema_labels = torch.max(ema_prob, dim=1)
                # ema_mask = 0
                # for i in range(sum(opts.num_classes[:-1])):
                #     ema_mask = ema_mask | ((labels <= fg_idx) & (ema_labels == i) & (ema_scores >= pseudo_thresh[i]))
                # ema_labels = torch.where(ema_mask, ema_labels, 255)

                outputs1, outputs2, outputs2_pos, outputs2_neg = model(images, pseudo_labels)

                # outputs1_kd = outputs1.permute(0, 2, 3, 1)[ema_labels != 255]
                # ema_scores_kd = ema_prob.permute(0, 2, 3, 1)[ema_labels != 255]
                # if opts.use_KD:
                #     kd_loss = criterion2(outputs[:, 1 + int(opts.unknown):-opts.num_classes[-1]],
                #                       outputs_prev.sigmoid()[:, 1 + int(opts.unknown):])
                #     supervision_loss = criterion(outputs, pseudo_labels)
                #     loss = 10 * kd_loss + supervision_loss
                #     #      10          1000

                if opts.train_branch == 'both' or opts.train_branch == 'ce':
                    loss_ce = criterion(outputs1, pseudo_labels)
                    loss_ema = dynamic_loss(outputs1, ema_prob, pseudo_thresh, labels)
                if opts.train_branch == 'both' or opts.train_branch == 'bce':
                    loss_bce = bceloss(outputs2[:, -opts.num_classes[-1]:], labels)
                    loss_kd = kdloss(outputs2[:, 1:sum(opts.num_classes[:-1])],
                                     outputs2_prev[:, 1:].sigmoid())
                    loss_ac = acloss(outputs2[:, 0:1], pseudo_labels)
                    loss_dkd_pos = kdloss(outputs2_pos[:, :-opts.num_classes[-1]],
                                          outputs2_pos_prev[:, :].sigmoid())
                    loss_dkd_neg = kdloss(outputs2_neg[:, :-opts.num_classes[-1]],
                                          outputs2_neg_prev[:, :].sigmoid())

            else:
                outputs1, outputs2, outputs2_pos, outputs2_neg = model(images, labels)
                if opts.train_branch == 'both' or opts.train_branch == 'ce':
                    loss_ce = criterion(outputs1, labels)
                if opts.train_branch == 'both' or opts.train_branch == 'bce':
                    if opts.unknown:
                        loss_bce = bceloss(outputs2[:, -opts.num_classes[-1] - 1:], labels)
                    else:
                        loss_bce = bceloss(outputs2[:, -opts.num_classes[-1]:], labels)
                    loss_ac = acloss(outputs2[:, 0:1], labels)

        loss = loss_ce + loss_bce + loss_ac + opts.kd * loss_dkd_pos + opts.dkd * loss_dkd_neg + opts.dkd * loss_kd \
               + opts.ema_loss * loss_ema
        scaler.scale(loss).backward()

        # print(outputs.shape,outputs.type(torch.float32).norm(p=1,dim=[0,2,3])/1024/1024/10)
        # should mul 16*16 because upsampling
        # print(model.module.classifier.head[-1][-1].weight.grad)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        if opts.curr_step > 0:
            model_ema.update_parameters(model)

        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        avg_celoss.update(loss_ce.item())
        avg_bceloss.update(loss_bce.item())
        avg_acloss.update(loss_ac.item())

        avg_kdloss.update(loss_kd.item())
        avg_dkdloss_pos.update(loss_dkd_pos.item())
        avg_dkdloss_neg.update(loss_dkd_neg.item())
        avg_emaloss.update(loss_ema.item())

        # if opts.use_KD and opts.curr_step > 0:
        #     avg_superloss.update(supervision_loss.item())
        #     avg_kdloss.update(kd_loss.item())

        if (cur_itrs) % opts.print_interval == 0:
            # if opts.use_KD and opts.curr_step > 0:
            #     print("[%s / step %d] Epoch %d, Itrs %d/%d, Super_loss=%6f, KD_loss=%6f, Total_Loss=%6f, Time=%.2f , LR=%.8f" %
            #           (opts.task, opts.curr_step, cur_epochs, cur_itrs, total_itrs,
            #            avg_superloss.avg, avg_kdloss.avg, avg_loss.avg, avg_time.avg*1000, optimizer.param_groups[0]['lr']))

            print("[%s / step %d] Epoch %d, Itrs %d/%d, Total_Loss=%6f, Time=%.2f , LR=%.8f" %
                  (opts.task, opts.curr_step, cur_epochs, cur_itrs, total_itrs,
                   avg_loss.avg, avg_time.avg * 1000, optimizer.param_groups[0]['lr']))

            print(f"Total loss ="
                  f"  ce_loss:{avg_celoss.avg:.2f}  +"
                  f"  bce_loss:{avg_bceloss.avg:.2f}  +"
                  f"  ac_loss:{avg_acloss.avg:.2f}  +"
                  f"  kd_loss:{avg_kdloss.avg:.2f}  +"
                  f"  dkd_loss_pos:{avg_dkdloss_pos.avg:.2f}  +"
                  f"  dkd_loss_neg:{avg_dkdloss_neg.avg:.2f}  +"
                  f"  ema_loss:{avg_emaloss.avg:.2f}")

        if opts.val_interval > 0 and (cur_itrs) % opts.val_interval == 0:
            print("validation...")
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader,
                                 device=device, metrics=metrics)
            print(metrics.to_str(val_score))

            model.train()

            class_iou = list(val_score['Class IoU'].values())
            curr_score = val_score["Mean IoU"]
            print("curr_val_score : %.4f" % (curr_score))
            print()

            if opts.curr_step > 0:
                val_ema_score = validate(opts=opts, model=model_ema, loader=val_loader,
                                         device=device, metrics=metrics)
                print(metrics.to_str(val_ema_score))
                curr_ema_score = val_ema_score["Mean IoU"]
                print("curr_ema_model_val_score : %.4f" % curr_ema_score)


            if curr_score > best_score:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))

    print("... Training Done")

    print("... Testing Best Model")
    best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)

    checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    model.module.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_score = validate(opts=opts, model=model, loader=test_loader,
                          device=device, metrics=metrics)
    print(metrics.to_str(test_score, out_matrix=True, opts=opts))

    # class_iou = list(test_score['Class IoU'].values())
    # class_acc = list(test_score['Class Acc'].values())
    # first_cls = len(get_tasks(opts.dataset, opts.task, 0))
    #
    # print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
    # print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
    # print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
    # print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':

    opts = get_argparser().parse_args()

    start_step = 0
    total_step = len(get_tasks(opts.dataset, opts.task))
    opts.now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    for step in range(start_step, total_step):
        opts.curr_step = step
        if step > 0:
            opts.lr = 0.001
        main(opts)
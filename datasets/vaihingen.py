import os
import sys
import torch.utils.data as data
import numpy as np
import json

import torch
from PIL import Image

from utils.tasks import get_dataset_list, get_tasks

classes = ('clutter', 'building', 'car', 'impervious_surface', 'low_vegetation', 'tree')

def vaihingen_cmap():
    cmap = np.zeros((16, 3), dtype=np.uint8)
    colors = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)


class VhSegmentation(data.Dataset):
    cmap = vaihingen_cmap()

    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None,
                 cil_step=0,
                 mem_size=0):

        self.root = opts.data_root
        self.task = opts.task
        self.overlap = opts.overlap
        self.unknown = opts.unknown

        self.image_set = image_set
        self.transform = transform

        dg_root = './datasets/data/vaihingen'

        # data root数据路径 != ade_root 数据txt路径
        if image_set == 'memory':
            image_dir = os.path.join(self.root, 'img_dir', 'train')
            mask_dir = os.path.join(self.root, 'ann_dir', 'train')
        elif image_set == 'test':
            image_dir = os.path.join(self.root, 'img_dir', 'val')
            mask_dir = os.path.join(self.root, 'ann_dir', 'val')
        else:
            image_dir = os.path.join(self.root, 'img_dir', image_set)
            mask_dir = os.path.join(self.root, 'ann_dir', image_set)

        assert os.path.exists(image_dir), "images not found"
        assert os.path.exists(mask_dir), "annotations not found"

        self.target_cls = get_tasks('vaihingen', self.task, cil_step)
        self.target_cls += [255]  # including ignore index (255)

        if image_set == 'test' or image_set == 'val' or image_set == 'test_val':
            for s in range(cil_step):
                self.target_cls += get_tasks('vaihingen', self.task, s)

        if image_set == 'test':
            file_names = open(os.path.join(dg_root, 'val_cls.txt'), 'r')
            file_names = file_names.read().splitlines()
            file_names = [line.split(' ')[0] for line in file_names]

        elif image_set == 'test_val':
            file_names = open(os.path.join(dg_root, 'val_cls.txt'), 'r')
            file_names = file_names.read().splitlines()
            file_names = [line.split(' ')[0] for line in file_names]

        elif image_set == 'memory':
            for s in range(cil_step):
                self.target_cls += get_tasks('vaihingen', self.task, s)

            memory_json = os.path.join(dg_root, 'memory.json')

            with open(memory_json, "r") as json_file:
                memory_list = json.load(json_file)

            file_names = memory_list[f"step_{cil_step}"]["memory_list"]
            print("... memory list : ", len(file_names), self.target_cls)

            while len(file_names) < opts.batch_size:
                file_names = file_names * 2

        else:
            file_names = get_dataset_list('vaihingen', self.task, cil_step, image_set, self.overlap)
            # 写一个新的txt
        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.masks = [os.path.join(mask_dir, x) for x in file_names]
        self.file_names = file_names

        # class re-ordering
        all_steps = get_tasks('vaihingen', self.task)  # [[],[],...]
        all_classes = []  # [0,1,2,...]
        for i in range(len(all_steps)):
            all_classes += all_steps[i]

        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        file_name = self.file_names[index]

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # sal_map is useless for ADE20K
        sal_map = Image.fromarray(np.ones(target.size[::-1], dtype=np.uint8))

        # re-define target label according to the CIL case
        target = self.gt_label_mapping(target)

        if self.transform is not None:
            img, target, sal_map = self.transform(img, target, sal_map)

        # add unknown label, background index: 0 -> 1, unknown index: 0
        if self.image_set == 'train' and self.unknown:
            target = torch.where(target == 255,
                                 torch.zeros_like(target) + 255,  # keep 255 (uint8)
                                 target + 1)  # unknown label

            unknown_area = (target == 1)
            # 全部背景都为1，转化为0 unknown，说明无背景
            target = torch.where(unknown_area, torch.zeros_like(target), target)

        return img, target.long(), sal_map, file_name

    def __len__(self):
        return len(self.images)

    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        gt = np.where(np.isin(gt, self.target_cls), gt, 0)

        if self.image_set == 'test' or self.image_set == 'val' or self.image_set == 'test_val':
            self.ordering_map[0] = 0

        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)

        return gt

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

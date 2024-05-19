import os
import cv2
import random
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, augment_prior, paired_random_crop_prior
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset, get_child_dirs
from basicsr.data.data_util import paired_paths_from_folder_prior
import re
import torch

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def natural_sort_key(s):
    """
    用于自然排序的key函数
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def unify_channels(tensor, c):
    """
    统一tensor的第4维通道数为32
    """
    channels, _, _ = tensor.size()
    if channels > c:
        # 超过c通道的截断
        tensor = tensor[:c, :, :]
    elif channels < c:
        # 不足c通道的补0
        zeros = torch.zeros((c - channels,) + tensor.size()[1:])
        tensor = torch.cat([tensor, zeros], dim=0)
    return tensor

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder, self.mask_grid1_folder, self.mask_grid2_folder, self.mask_grid3_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask_grid1'], \
                                                                                                                 opt['dataroot_mask_grid2'], opt['dataroot_mask_grid3']
                
        self.lq_paths = make_dataset(self.lq_folder)
        self.gt_paths = make_dataset(self.gt_folder)
        self.mask_grid1_folder = get_child_dirs(self.mask_grid1_folder)
        self.mask_grid2_folder = get_child_dirs(self.mask_grid2_folder)
        self.mask_grid3_folder = get_child_dirs(self.mask_grid3_folder)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        #  scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.
        lq_path = self.lq_paths[index]
        img_lq = cv2.imread(lq_path).astype(np.float32) / 255.

        # mask_grid1_path = self.mask_grid1_folder[index]
        # img_mask_grid11 = []
        # for mask in sorted(os.listdir(mask_grid1_path), key=natural_sort_key):
        #     mask_path = os.path.join(mask_grid1_path, mask)
        #     img_mask_grid1 = cv2.imread(mask_path).astype(np.float32) / 255.
        #     img_mask_grid11.append(img_mask_grid1)
        #
        # mask_grid2_path = self.mask_grid2_folder[index]
        # img_mask_grid22 = []
        # for mask in sorted(os.listdir(mask_grid2_path), key=natural_sort_key):
        #     mask_path = os.path.join(mask_grid2_path, mask)
        #     img_mask_grid2 = cv2.imread(mask_path).astype(np.float32) / 255.
        #     img_mask_grid22.append(img_mask_grid2)
        #
        # mask_grid3_path = self.mask_grid3_folder[index]
        # img_mask_grid33 = []
        # for mask in sorted(os.listdir(mask_grid3_path), key=natural_sort_key):
        #     mask_path = os.path.join(mask_grid3_path, mask)
        #     img_mask_grid3 = cv2.imread(mask_path).astype(np.float32) / 255.
        #     img_mask_grid33.append(img_mask_grid3)

        # augmentation for training
        if self.opt['phase'] == 'train':
            input_gt_size = img_gt.shape[0]
            input_lq_size = img_lq.shape[0]
            scale = input_gt_size // input_lq_size
            gt_size = self.opt['gt_size']

            if self.opt['use_resize_crop']:
                # random resize
                input_gt_random_size = random.randint(gt_size, input_gt_size)
                input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale
                resize_factor = input_gt_random_size / input_gt_size
                img_gt = random_resize(img_gt, resize_factor)
                img_lq = random_resize(img_lq, resize_factor)

                # img_mask_grid_r11 = []
                # for mask in img_mask_grid11:
                #     img_mask_grid_r1 = random_resize(mask, resize_factor)
                #     img_mask_grid_r11.append(img_mask_grid_r1)
                # img_mask_grid11 = img_mask_grid_r11
                #
                # img_mask_grid_r22 = []
                # for mask in img_mask_grid22:
                #     img_mask_grid_r2 = random_resize(mask, resize_factor)
                #     img_mask_grid_r22.append(img_mask_grid_r2)
                # img_mask_grid22 = img_mask_grid_r22
                #
                # img_mask_grid_r33 = []
                # for mask in img_mask_grid33:
                #     img_mask_grid_r3 = random_resize(mask, resize_factor)
                #     img_mask_grid_r33.append(img_mask_grid_r3)
                # img_mask_grid33 = img_mask_grid_r33

                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, input_gt_size // input_lq_size, gt_path)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

                # random crop
                # img_gt, img_lq, img_mask_grid11, img_mask_grid22, img_mask_grid33 = paired_random_crop_prior(img_gt, img_lq, img_mask_grid11, img_mask_grid22, img_mask_grid33, gt_size, input_gt_size // input_lq_size, gt_path)

            # flip, rotation
            # img_gt, img_lq, img_mask_grid11, img_mask_grid22, img_mask_grid33 = augment_prior([img_gt, img_lq, img_mask_grid11, img_mask_grid22, img_mask_grid33], self.opt['use_flip'], self.opt['use_rot'])

            # cv2.imshow('img_gt', img_gt)
            # cv2.imshow('img_lq', img_lq)
            # cv2.imshow('img_mask_grid11', img_mask_grid11[0])
            # cv2.imshow('img_mask_grid22', img_mask_grid22[0])
            # cv2.imshow('img_mask_grid33', img_mask_grid33[0])
            # cv2.waitKey(0)

        if self.opt['phase'] != 'train':
            crop_eval_size = self.opt.get('crop_eval_size', None)
            if crop_eval_size:
                input_gt_size = img_gt.shape[0]
                input_lq_size = img_lq.shape[0]
                scale = input_gt_size // input_lq_size
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, crop_eval_size, input_gt_size // input_lq_size,
                                               gt_path)

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # img_mask_grid111 = []
        # for img_mask in img_mask_grid11:
        #     img_mask1 = img2tensor(img_mask, bgr2rgb=True, float32=True)
        #     img_mask_grid111.append(img_mask1)
        #
        # img_mask_grid222 = []
        # for img_mask in img_mask_grid22:
        #     img_mask2 = img2tensor(img_mask, bgr2rgb=True, float32=True)
        #     img_mask_grid222.append(img_mask2)
        #
        # img_mask_grid333 = []
        # for img_mask in img_mask_grid33:
        #     img_mask3 = img2tensor(img_mask, bgr2rgb=True, float32=True)
        #     img_mask_grid333.append(img_mask3)
        #
        # img_mask_grid111 = torch.cat(img_mask_grid111, 0)
        # img_mask_grid222 = torch.cat(img_mask_grid222, 0)
        # img_mask_grid333 = torch.cat(img_mask_grid333, 0)
        #
        # img_mask_grid111 = unify_channels(img_mask_grid111, 64)
        # img_mask_grid222 = unify_channels(img_mask_grid222, 128)
        # img_mask_grid333 = unify_channels(img_mask_grid333, 256)

        # return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'mask_grid1': img_mask_grid111, 'mask_grid2': img_mask_grid222, 'mask_grid3': img_mask_grid333}
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

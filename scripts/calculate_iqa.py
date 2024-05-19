import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
import glob
from basicsr.utils import scandir

import pyiqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='/media/HDD1/yj/yj_LLIE/SeD_nopre/VV_SR/')
    parser.add_argument('--gt_path', type=str, default='/media/HDD1/yj/yj_LLIE/SeD_nopre/VV_SR/')
    parser.add_argument('--metrics', nargs='+', default=['musiq', 'nrqm', 'niqe'])
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces')

    args = parser.parse_args()

    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    if args.gt_path.endswith('/'):  # solve when path ends with /
        args.gt_path = args.gt_path[:-1]

    # Initialize metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iqa_musiq, iqa_nrqm, iqa_niqe = None, None, None
    score_musiq_all, score_nrqm_all, score_niqe_all = [], [], []
    print(args.metrics)
    if 'musiq' in args.metrics:
      iqa_musiq = pyiqa.create_metric('musiq').to(device)
      iqa_musiq.eval()
    if 'nrqm' in args.metrics:
      iqa_nrqm = pyiqa.create_metric('nrqm').to(device)
      iqa_nrqm.eval()
    if 'niqe' in args.metrics:
      iqa_niqe = pyiqa.create_metric('niqe').to(device)
      iqa_niqe.eval()

    img_out_paths = sorted(list(scandir(args.result_path, suffix=('jpg', 'png', 'bmp'),
                                    recursive=True, full_path=True)))
    total_num = len(img_out_paths)

    for i, img_out_path in enumerate(img_out_paths):
        img_name = img_out_path.replace(args.result_path+'/', '')
        cur_i = i + 1
        print(f'[{cur_i}/{total_num}] Processing: {img_name}')
        img_out = cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()
        try:
          img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
          img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
          img_gt = np.transpose(img_gt, (2, 0, 1))
          img_gt = torch.from_numpy(img_gt).float()
          with torch.no_grad():
            img_out = img_out.unsqueeze(0).to(device)
            img_gt = img_gt.unsqueeze(0).to(device)
            if iqa_niqe is not None:
              score_niqe_all.append(iqa_niqe(img_out, img_gt).item())



        except:
          print(f'skip: {img_name}')
          continue
        if (i+1)%20 == 0:
          print(f'[{cur_i}/{total_num}] NIQE: {sum(score_niqe_all)/len(score_niqe_all)},\n')

    print('-------------------Final Scores-------------------\n')
    print(f' NIQE: {sum(score_niqe_all)/len(score_niqe_all)}')

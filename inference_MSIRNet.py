import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load
import numpy as np

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.lesnet17_arch import LESNet17
from basicsr.utils.download_util import load_file_from_url 
from thop import profile
from thop import clever_format

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ### cuda

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}


def unify_channels(tensor, c):
    """
    统一tensor的第4维通道数为32
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels, _, _ = tensor.size()
    if channels > c:
        # 超过c通道的截断
        tensor = tensor[:c, :, :]
    elif channels < c:
        # 不足c通道的补0
        zeros = torch.zeros((c - channels,) + tensor.size()[1:]).to(device)
        tensor = torch.cat([tensor, zeros], dim=0)
    return tensor
    

def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='dataset/LOL_SR/test/lowX4/', help='Input image or folder')
    parser.add_argument('-g8', '--grid8', type=str, default='dataset/LOL_SR/test/lowX4_HE_BM3D_masks/grid8/', help='Input image or folder')
    parser.add_argument('-g16', '--grid16', type=str, default='dataset/LOL_SR/test/lowX4_HE_BM3D_masks/grid16/', help='Input image or folder')
    parser.add_argument('-g32', '--grid32', type=str, default='dataset/LOL_SR/test/lowX4_HE_BM3D_masks/grid32/', help='Input image or folder')

    parser.add_argument('-w', '--weight', type=str, default='/media/HDD1/yj/yj_LLIE/MSIR/experiments/net_g_111250.pth', help='path for model weights') ##
    parser.add_argument('-o', '--output', type=str, default='results-MSIRNet/', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image') ##
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=6250, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    if args.weight is None:
        weight_path = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    else:
        weight_path = args.weight
    
    # set up the model
    sr_model = LESNet17(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=args.out_scale).to(device)  ## model
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.  # 3,625,625
        img_tensor = img_tensor.unsqueeze(0)  # 1,3,625,625


        #####____________prepair for sam_masks___________#####
        img_name_base, _ = os.path.splitext(img_name)
        file_name8 = os.path.join(args.grid8, img_name_base)
        img_mask_grid11 = []
        for mask in sorted(os.listdir(file_name8)):
            mask_path = os.path.join(file_name8, mask)
            img_mask_grid1 = cv2.imread(mask_path).astype(np.float32) / 255.
            img_mask1 = img2tensor(img_mask_grid1, bgr2rgb=True, float32=True).to(device)
            img_mask_grid11.append(img_mask1)

        file_name16 = os.path.join(args.grid16, img_name_base)
        img_mask_grid22 = []
        for mask in sorted(os.listdir(file_name16)):
            mask_path = os.path.join(file_name16, mask)
            img_mask_grid2 = cv2.imread(mask_path).astype(np.float32) / 255.
            img_mask2 = img2tensor(img_mask_grid2, bgr2rgb=True, float32=True).to(device)
            img_mask_grid22.append(img_mask2)

        file_name32 = os.path.join(args.grid32, img_name_base)
        img_mask_grid33 = []
        for mask in sorted(os.listdir(file_name32)):
            mask_path = os.path.join(file_name32, mask)
            img_mask_grid3 = cv2.imread(mask_path).astype(np.float32) / 255.
            img_mask3 = img2tensor(img_mask_grid3, bgr2rgb=True, float32=True).to(device)
            img_mask_grid33.append(img_mask3)

        img_mask_grid111 = torch.cat(img_mask_grid11, 0)
        img_mask_grid222 = torch.cat(img_mask_grid22, 0)
        img_mask_grid333 = torch.cat(img_mask_grid33, 0)

        img_mask_grid111 = unify_channels(img_mask_grid111, 64)
        img_mask_grid222 = unify_channels(img_mask_grid222, 128)
        img_mask_grid333 = unify_channels(img_mask_grid333, 256)

        img_mask_grid111 = img_mask_grid111.unsqueeze(0)
        img_mask_grid222 = img_mask_grid222.unsqueeze(0)
        img_mask_grid333 = img_mask_grid333.unsqueeze(0)

        ############ pytorch calculation params ###########
        # print(img_tensor.shape)
        # test_img = torch.rand((1, 3, 256, 256)).cuda()
        # print(test_img.shape)
        # macs, params = profile(sr_model, inputs=(test_img,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("Model FLOPs: ", macs)
        # print("Model Params:", params)

        max_size = args.max_size ** 2 
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output = sr_model.test(img_tensor, img_mask_grid111, img_mask_grid222, img_mask_grid333)
        else:
            output = sr_model.test_tile(img_tensor, img_mask_grid111, img_mask_grid222, img_mask_grid333)
        output_img = tensor2img(output)

        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    main()

import argparse
import random
from argparse import ArgumentParser

import os
import sys
import numpy as np
import datetime
import PIL.Image as Image
import json  # [新增] 引入 json 模块
import shutil # [新增] 用于复制不需要增强的原图

import logging
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import autocast, GradScaler

# 如果不需要 adamp 和 pyiqa 可以注释掉，保持原样
from adamp import AdamP
from itertools import cycle
import pyiqa

from collections import OrderedDict
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))
    return device, available_gpus

def get_args():
    parser = argparse.ArgumentParser(description='Evaliation')

    ### >>> [data] set data info.
    parser.add_argument('--database', default='DTIUIE', type=str, help='database name')
    parser.add_argument('--crop_size', type=int, default=256, help='input image size for training (default: 256)')

    ### >>> [model] set model info.
    parser.add_argument('--model_E', default='DTIUIE', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='path to latest checkpoint (default: None)')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('-g', '--gpus', type=int, default=1, metavar='N')

    parser.add_argument('--draw_images', action='store_true', default=True, help='flag whether to draw images')
    parser.add_argument('--save_dir', default='/root/autodl-tmp/bi/DTIUIE_DUO', type=str, help='path to save images')#!!!
    
    # [新增] JSON 文件路径参数
    parser.add_argument('--json_path', default='/root/autodl-tmp/bi/DUO_step2_fixed.json', type=str, help='path to VLM evaluation json') #!!!

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.local_rank = -1

    device, available_gpus = get_available_devices(args.gpus)
    
    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # ================= [新增] 1. 读取 JSON 建立增强决策字典 =================
    print(f"读取诊断结果: {args.json_path}")
    enhancement_map = {}
    if os.path.exists(args.json_path):
        with open(args.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for item in json_data:
                img_path = item.get('image_path', '')
                img_name = os.path.basename(img_path)
                # 获取是否需要增强的布尔值 (默认设为 True 以防解析失败)
                needs_enh = item.get('hybrid_evaluation', {}).get('needs_enhancement', True)
                enhancement_map[img_name] = needs_enh
        print(f"成功加载 {len(enhancement_map)} 张图片的诊断结果。")
    else:
        print("⚠️ 警告: 未找到 JSON 文件，所有图片将默认进行增强处理！")
    # =======================================================================

    # >>> [model] set model info.
    if  args.model_E == 'DTIUIE':
        from models.enhancement.model_dtiuie import DTIUIE
        # [修复] 修正了实例化名字 TIUIED
        model_E = DTIUIE()
        model_E = model_E.to(device)
        args.pretrained = './checkpoints/dtiuie_ckpt.pth'
    
    from models.segmentation.vgg_unet import VGG16Unet
    model_S = VGG16Unet(n_channels=3, n_classes=8, pretrained=True)
    model_S = model_S.to(device)

    from dataloader_seg import Tested_Set
    dataset_dir = '/root/autodl-tmp/bi/DUO'#!!!
    args.classes = 8
    dataset = Tested_Set(file_path=dataset_dir, status='valid', augmentation=False, angle=0, size_h=args.crop_size, size_w=args.crop_size, hflip_p=0)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=8, pin_memory=True)

    if args.pretrained:
        pre_trained_state_dict = torch.load(args.pretrained, map_location=device)
        model_E.load_state_dict(pre_trained_state_dict['model_E'])
        model_S.load_state_dict(pre_trained_state_dict['model_S'])
        
    model_E.to(device=device)
    model_S.to(device=device)
    model_E.eval()
    model_S.eval()
    
    num_val_batches = len(dataloader)

    # with torch.no_grad():
    #     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', ncols=150, leave=True):
            
    #         images_names = batch['image_name']
    #         current_img_name = os.path.basename(images_names[0]) # batch_size 是 1
            
    #         # ================= [新增] 2. 检查是否需要增强 =================
    #         needs_enhancement = enhancement_map.get(current_img_name, True)
            
    #         if not needs_enhancement:
    #             # 如果不需要增强，为了后续流程连贯，我们直接把原图复制到输出文件夹
    #             # 这样下一步目标检测时，就只用去 ./outputs/DTIUIE 文件夹下读取即可
    #             original_img_path = images_names[0]
    #             save_path = os.path.join(args.save_dir, current_img_name)
                
    #             # 如果你想节省空间，也可以注释掉下面两行，目标检测时再根据 JSON 切换路径
    #             # if os.path.exists(original_img_path):
    #             #     shutil.copy(original_img_path, save_path)
                    
    #             continue # 🔥 核心逻辑：跳过下方的神经网络前向传播，极大节省时间！
    #         # =============================================================

    #         # --- 下面是原有的图像增强网络逻辑 ---
    #         images = batch['image'].to(device)
    #         _, _, original_h, original_w = images.shape
    #         new_h = (original_h + 31) // 32 * 32 
    #         new_w = (original_w + 31) // 32 * 32 
    #         if original_h * original_w > 1088 * 1920:
    #             new_h = 1088
    #             new_w = 1920
    #         images = F.interpolate(images, size=(new_h, new_w), mode='bicubic', align_corners=False)
            
    #         # 注意：此处您原代码强制将 output 的保存尺寸设回了 256x256
    #         original_h, original_w = 256, 256 
    #         images = F.interpolate(images, size=(256, 256), mode='bicubic', align_corners=False)

    #         # Generate output
    #         _, feat_raw = model_S(images, return_feats=True)
    #         outputs = model_E(images, feat_raw)

    #         if args.draw_images:
    #             for i, (output, image_name) in enumerate(zip(outputs, images_names)):
    #                 output = output.unsqueeze(0)
    #                 output = F.interpolate(output, size=(original_h, original_w), mode='bicubic', align_corners=False)
    #                 output = output.squeeze(0)

    #                 output = output.permute(1, 2, 0).cpu().numpy()
    #                 output = (output * 255).astype(np.uint8)
    #                 output = Image.fromarray(output)
                    
    #                 save_path = os.path.join(args.save_dir, os.path.basename(image_name))
    #                 output.save(save_path)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', ncols=150, leave=True):
            
            images_names = batch['image_name']
            current_img_path = images_names[0]
            current_img_name = os.path.basename(current_img_path)
            
            # [1] 获取原图的真实宽高 (防止被 dataloader 的 crop_size 限制)
            temp_img = Image.open(current_img_path)
            true_w, true_h = temp_img.size 

            # [新增] 2. 检查是否需要增强
            needs_enhancement = enhancement_map.get(current_img_name, True)
            
            if not needs_enhancement:
                save_path = os.path.join(args.save_dir, current_img_name)
                if os.path.exists(current_img_path):
                    shutil.copy(current_img_path, save_path)
                continue

            # --- 增强逻辑 ---
            images = batch['image'].to(device)
            
            # [2] 这一部分是为了适配模型输入，保持 256 进行计算
            # 但不要用 original_h/w 命名，以免混淆
            model_input_size = (256, 256)
            input_images = F.interpolate(images, size=model_input_size, mode='bicubic', align_corners=False)

            # Generate output
            _, feat_raw = model_S(input_images, return_feats=True)
            outputs = model_E(input_images, feat_raw)

            if args.draw_images:
                for i, (output, image_name) in enumerate(zip(outputs, images_names)):
                    output = output.unsqueeze(0)
                    
                    # [3] 核心修改：插值回真正的原图尺寸 (true_h, true_w)
                    output = F.interpolate(output, size=(true_h, true_w), mode='bicubic', align_corners=False)
                    
                    output = output.squeeze(0)
                    output = output.permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).clip(0, 255).astype(np.uint8) # 加上 clip 防止溢出
                    output = Image.fromarray(output)
                    
                    save_path = os.path.join(args.save_dir, os.path.basename(image_name))
                    output.save(save_path)
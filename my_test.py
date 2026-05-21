import sys
import os
import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
# 确保能找到同级目录下的模块
from dataloader import UJEDDataset
from model.model import DGU_Net
from utils.data_utils import collate_fn
from ultralytics import YOLO
from torch.cuda.amp import autocast
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Testing enhancement model')
    
    # =====================================================================
    # 路径配置区 (已根据你的需求硬编码)
    # =====================================================================
    
    # 1. 测试数据集路径
    parser.add_argument('--test_dir', type=str, 
                        default='/root/autodl-tmp/bi/DGUNet/datasets/pre_color', 
                        help='Test dataset path')
    
    # 2. 增强后图片的保存路径
    parser.add_argument('--save_dir', type=str, 
                        default='/root/autodl-tmp/bi/DGUNet/results/enhanced_color', 
                        help='Directory to save enhanced images')
    
    # 3. 增强模型 (DGU_Net) 的权重路径
    parser.add_argument('--en_weight', type=str, 
                        default='/root/autodl-tmp/bi/DGUNet/weights/enhance_model.pth', 
                        help='Path to trained enhancement model')
    
    # 4. YOLO 检测先验的权重路径
    parser.add_argument('--det_weight', type=str, 
                        default='/root/autodl-tmp/bi/DGUNet/weights/detection_prior.pt', 
                        help='Path to trained detection prior model')

    # =====================================================================
    
    # 【关键修改】为了保持原图分辨率且不报错，batch_size 强制默认为 1
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size must be 1 for varying original sizes')
    
    # 此处的 image_size 仅作保留，由于下面没有使用 Resize，它不会影响出图尺寸
    parser.add_argument('--image_size', type=int, default=256, help='Image size (Not used when keeping original size)')
    parser.add_argument('--n_block', type=int, default=3, help='Number of Dual_Net blocks')
    
    args = parser.parse_args()
    return args

def main(args):
    # 1. 定义你的“散装”路径
    path_config = {
        'images': '/root/autodl-tmp/bi/ROUD/color',          # 原图路径
        't_prior': '/root/autodl-tmp/bi/DGUNet/pre_color/t_prior', # 预处理t图
        'B_prior': '/root/autodl-tmp/bi/DGUNet/pre_color/B_prior', # 预处理B图
        'labels': '/root/autodl-tmp/bi/DGUNet/pre_color/labels'    # 标签路径
    }
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"📁 已创建结果保存目录: {args.save_dir}")
    # 2. 检查这些路径是否存在
    for k, v in path_config.items():
        if not os.path.exists(v):
            print(f"❌ 找不到路径: {k} -> {v}")
            return

    transform = [transforms.ToTensor()]
    
    # 3. 将字典传给 Dataset
    test_dataset = UJEDDataset(path_config, transform=transform, is_test=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, # 强制为1
        shuffle=False,
        num_workers=4, 
        collate_fn=collate_fn
    )
    print(f"[INFO] 成功加载 {len(test_dataset)} 张测试图片.")

    # ================= 3. 模型加载 =================
    print("Loading Enhance Net...")
    model = DGU_Net(Block_number=args.n_block).to(DEVICE)
    checkpoint = torch.load(args.en_weight, map_location=DEVICE)
    
    # 兼容性加载：处理模型权重字典的结构差异
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Loading YOLO Det Net...")
    det_api = YOLO(args.det_weight)
    det_model = det_api.model.eval()      
    for p in det_model.parameters():
        p.requires_grad_(False) # 冻结检测模型，节约显存
    det_model.to(DEVICE)
    
    # ================= 4. 开始推理 =================
    print("\n🚀 开始保持原图分辨率推理...")
    with torch.no_grad(): # 禁用梯度，大幅节约显存
        for batch in tqdm(test_loader, desc="Processing"):
            input_img = batch["inp"].to(DEVICE)   
            t_p = batch["t"].to(DEVICE)
            B_p = batch["B"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)   
            fn_list = batch["fn"]

            # 如果透射率图是单通道，扩展为三通道以匹配模型输入
            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1, 3, 1, 1)
            # if input_img.shape[2] * input_img.shape[3] > 1500000:
            #     scale = 0.5
            #     input_img = F.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
            #     t_p = F.interpolate(t_p, scale_factor=scale, mode='bilinear', align_corners=False)
            #     B_p = F.interpolate(B_p, scale_factor=scale, mode='bilinear', align_corners=False)
            h, w = input_img.shape[2], input_img.shape[3]
            max_side = max(h, w)
        
        # 只要长边超过 1280 (标准 720P 的水平)，就强制按比例缩放
            if max_side > 1280:
                scale = 1280 / max_side
                input_img = F.interpolate(input_img, scale_factor=scale, mode='bilinear')
                t_p = F.interpolate(t_p, scale_factor=scale, mode='bilinear')
                B_p = F.interpolate(B_p, scale_factor=scale, mode='bilinear')
                print(f" -> 发现超长图 {fn_list[0]} ({w}x{h}), 强制等比缩放到长边 1280")
            try:
                # 1. 记录原图的原始高(h)和宽(w)
                h, w = input_img.shape[2], input_img.shape[3]
                
                # 2. 计算需要填充的数值 (补齐到 16 的倍数)
                pad_factor = 16 
                pad_h = (pad_factor - h % pad_factor) % pad_factor
                pad_w = (pad_factor - w % pad_factor) % pad_factor
                
                # 3. 执行填充 (在右侧和下方填充)
                # 使用 'reflect' 镜像填充效果比黑边更好，能减少边缘伪影
                if pad_h > 0 or pad_w > 0:
                    input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')
                    t_p = F.pad(t_p, (0, pad_w, 0, pad_h), mode='reflect')
                    B_p = F.pad(B_p, (0, pad_w, 0, pad_h), mode='reflect')
                    # 如果你的 labels 也是 Tensor 且参与了模型运算，也需要对其进行填充
                    # labels = F.pad(labels, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                # 4. 模型前向传播
                output_J, _, _ = model(input_img, t_p, B_p, labels, det_api)
                enhanced_img = output_J[-1] 
                
                # 5. 【关键】将结果裁剪回原图尺寸
                if pad_h > 0 or pad_w > 0:
                    enhanced_img = enhanced_img[:, :, :h, :w]
                # # 模型前向传播
                # output_J, _, _ = model(input_img, t_p, B_p, labels, det_api)
                # enhanced_img = output_J[-1] # 取最后一层的输出作为最终结果

                # 保存图片
                for idx in range(enhanced_img.size(0)):
                    fn = fn_list[idx]
                    save_path = os.path.join(args.save_dir, fn)
                    # 使用 clamp 限制像素值在 0-1 之间，防止噪点过曝
                    save_image(enhanced_img[idx].clamp(0, 1), save_path)
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ 处理 {fn_list[0]} 时显存溢出！原图分辨率可能过大。")
                    torch.cuda.empty_cache() # 尝试清空显存继续下一张
                else:
                    print(f"\n❌ 处理 {fn_list[0]} 时发生错误: {e}")
                    print("提示：请确认原图长宽是否均为 8 的倍数（UNet结构要求）。")

    print("\n🎉 [DONE] 所有图片已处理完成！")
    print(f"原图分辨率增强结果请前往 {args.save_dir} 查看。")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
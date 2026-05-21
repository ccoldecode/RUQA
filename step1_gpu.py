import cv2
import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing as mp

# ==================== 关键修复：设置 spawn + 延迟 CUDA 初始化 ====================
# 在 if __name__ == '__main__' 之前不要创建任何 CUDA tensor 或 device

def get_device():
    """每个子进程独立获取 device，避免 fork 问题"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== GPU 加速的 EME 函数（每个进程独立使用） ====================
def torch_eme(img_tensor: torch.Tensor, block_size: int = 8) -> float:
    if img_tensor.device.type == 'cpu':
        img_tensor = img_tensor.to(get_device())
    
    B, C, H, W = img_tensor.shape
    h_new = (H // block_size) * block_size
    w_new = (W // block_size) * block_size
    if h_new == 0 or w_new == 0:
        return 0.0
    
    img = img_tensor[:, :, :h_new, :w_new]
    blocks = img.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    blocks = blocks.contiguous().view(-1, block_size * block_size)
    
    bmin = blocks.min(dim=1)[0]
    bmax = blocks.max(dim=1)[0]
    bmin = torch.clamp(bmin, min=1.0)
    bmax = torch.clamp(bmax, min=1.0)
    
    eme_vals = 20.0 * torch.log10(bmax / bmin)
    return float(eme_vals.mean().item())


# ==================== 单张图像处理（GPU 版） ====================
def compute_all_metrics_gpu(img_path):
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None

        # 转 Tensor（CPU 准备）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        device = get_device()

        # ==================== 修正后的 UCIQE（值正常 0.3~0.8） ====================
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l = img_lab[:,:,0].astype(np.float32) / 255.0          # L 归一化到 0-1
        a = img_lab[:,:,1].astype(np.float32) / 255.0
        b = img_lab[:,:,2].astype(np.float32) / 255.0
        chroma = np.sqrt(a**2 + b**2)

        # 1. chroma variance（主流论文实现）
        aver_chr = np.mean(chroma)
        var_chr = np.sqrt(np.mean(np.abs(1 - (aver_chr / chroma)**2)))

        # 2. contrast（L 0-1 下的 1% 对比度）
        l_flat = l.flatten()
        top = max(1, int(np.round(0.01 * len(l_flat))))
        sl = np.sort(l_flat)
        conl = np.mean(sl[-top:]) - np.mean(sl[:top])

        # 3. saturation
        sat = chroma / np.sqrt(chroma**2 + l**2 + 1e-8)
        us = np.mean(sat)

        uciqe = 0.4680 * var_chr + 0.2745 * conl + 0.2576 * us

        # ==================== UIQM（已修复 yb_tr bug） ====================
        # UICM（trimmed mean + 强保护）
        rg = img_rgb[:, :, 0] - img_rgb[:, :, 1]
        yb = 0.5 * (img_rgb[:, :, 0] + img_rgb[:, :, 1]) - img_rgb[:, :, 2]
        rgl = np.sort(rg.flatten())
        ybl = np.sort(yb.flatten())
        
        num_pixels = len(rgl)
        T = max(1, int(0.1 * num_pixels)) if num_pixels > 0 else 0
        
        rg_tr = rgl[T:-T] if num_pixels > 2 * T else rgl
        yb_tr = ybl[T:-T] if num_pixels > 2 * T else ybl   # ← 修复点在这里
        
        uicm = -0.0268 * np.sqrt(np.mean(rg_tr)**2 + np.mean(yb_tr)**2) + \
               0.1586 * np.sqrt(np.var(rg_tr) + np.var(yb_tr))

        # UISM（GPU Sobel + EME）
        img_tensor = img_tensor.to(device)
        def sobel_edge_t(t):
            sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
            sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3)
            gx = torch.nn.functional.conv2d(t, sobel_x, padding=1)
            gy = torch.nn.functional.conv2d(t, sobel_y, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)

        r_edge = sobel_edge_t(img_tensor[:,0:1]) * img_tensor[:,0:1]
        g_edge = sobel_edge_t(img_tensor[:,1:2]) * img_tensor[:,1:2]
        b_edge = sobel_edge_t(img_tensor[:,2:3]) * img_tensor[:,2:3]

        Reme = torch_eme(r_edge)
        Geme = torch_eme(g_edge)
        Beme = torch_eme(b_edge)
        uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

        # UIConM
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
        uiconm = torch_eme(gray_tensor)

        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm

        # 其他指标
        means = [float(img_rgb[:,:,i].mean()) for i in range(3)]
        color_imb = (max(means) - min(means)) / (sum(means)/3 + 1e-8)
        brightness = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
        lap_var = cv2.Laplacian(cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (512,512)), cv2.CV_64F).var()

        return {
            "image_name": img_path.name,
            "image_path": str(img_path),
            "metrics": {
                "uciqe": round(float(uciqe), 4),
                "uiqm": round(float(uiqm), 4),
                "color_imbalance": round(float(color_imb), 4),
                "brightness": round(float(brightness), 2),
                "laplacian_var": round(float(lap_var), 2)
            }
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
# ==================== 主程序（关键修改在这里） ====================
def main():
    input_dir = "/root/autodl-tmp/bi/DUO"
    output_file = "DUO_step1.json"

    path_obj = Path(input_dir)
    img_list = list(path_obj.glob("*.jpg")) + list(path_obj.glob("*.png"))

    num_workers = min(os.cpu_count(), 8)   # spawn 模式开销较大，建议先用 6~10

    print(f"开始处理: {len(img_list)} 张图像 | 进程数: {num_workers} | 启动方法: spawn")

    # 使用 spawn 上下文（最重要的一行）
    mp_context = mp.get_context('spawn')
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        for res in tqdm(executor.map(compute_all_metrics_gpu, img_list, chunksize=10), total=len(img_list)):
            if res:
                results.append(res)

    print(f"正在保存结果到 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成！共计保存 {len(results)} 条数据。")


if __name__ == "__main__":
    # 在这里设置 spawn 是安全的
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
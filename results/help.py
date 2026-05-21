import json
import os
import cv2
import random
import numpy as np

# ================= 配置区 =================
GT_JSON = "/root/autodl-tmp/bi/results/instances_color.json"
RES_JSON = "/root/autodl-tmp/bi/ROUD_step4_color_wE.json"
# IMG_DIR = "/root/autodl-tmp/bi/ROUD/blur" # 你的原始图片路径
IMG_DIR = "/root/autodl-tmp/bi/DTIUIE_ROUD_color" # 你的原始图片路径
SAVE_DIR = "/root/autodl-tmp/bi/ROUD_visual" # 可视化结果保存路径

def visualize_alignment(num_samples=5):
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    with open(GT_JSON, 'r') as f: gt_data = json.load(f)
    with open(RES_JSON, 'r') as f: res_data = json.load(f)

    # 1. 建立映射
    gt_imgs = {img['id']: img for img in gt_data['images']}
    res_imgs = {img['id']: img for img in res_data['images']}
    
    # 按图片名归类标注
    gt_annos = {}
    for ann in gt_data['annotations']:
        fname = gt_imgs[ann['image_id']]['file_name']
        gt_annos.setdefault(fname, []).append(ann)
        
    res_annos = {}
    for ann in res_data['annotations']:
        fname = res_imgs[ann['image_id']]['file_name']
        res_annos.setdefault(fname, []).append(ann)

    # 2. 随机选几张有预测结果的图
    sample_fnames = random.sample(list(res_annos.keys()), min(num_samples, len(res_annos)))

    print(f"🧐 正在生成 {len(sample_fnames)} 张对比图到 {SAVE_DIR}...")

    for fname in sample_fnames:
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳过: 找不到图片 {img_path}")
            continue

        # 画真值 (GT) - 绿色
        for ann in gt_annos.get(fname, []):
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"GT_{ann['category_id']}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 画预测 (Pred) - 红色
        for ann in res_annos.get(fname, []):
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f"P_{ann['category_id']}", (x, y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(SAVE_DIR, f"compare_{fname}"), img)
        
        # 打印这条信息帮你人肉对齐
        print(f"\n🖼️ 文件: {fname}")
        print(f"   GT ID(官方): {next(i['id'] for i in gt_data['images'] if i['file_name']==fname)}")
        print(f"   你的 ID: {next(i['id'] for i in res_data['images'] if i['file_name']==fname)}")
        print(f"   标注数量: GT={len(gt_annos.get(fname, []))}, Pred={len(res_annos.get(fname, []))}")

if __name__ == "__main__":
    visualize_alignment()
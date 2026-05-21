# import json
# import os
# from PIL import Image, ImageDraw, ImageFont
# from tqdm import tqdm

# # ================= 配置 =================
# JSON_PATH = "/root/autodl-tmp/bi/step4_qwen_raw_50_test_v2.json"
# IMAGE_DIR = "/root/autodl-tmp/bi/data/dataset/"
# VIS_DIR = "/root/autodl-tmp/bi/step4_vis/"

# # 类别颜色
# COLORS = {
#     "sea cucumber": "#FF0000", "sea urchin": "#00FF00", "scallop": "#0000FF",
#     "starfish": "#FFFF00", "fish": "#FF00FF", "corals": "#00FFFF",
#     "diver": "#FFA500", "squid": "#800080", "turtle": "#008000",
#     "jellyfish": "#FF69B4"
# }

# os.makedirs(VIS_DIR, exist_ok=True)

# # 加载 COCO
# with open(JSON_PATH, 'r', encoding='utf-8') as f:
#     coco = json.load(f)

# image_map = {img["id"]: img for img in coco["images"]}

# ann_by_image = {}
# for ann in coco["annotations"]:
    
#     img_id = ann["image_id"]
#     if img_id not in ann_by_image:
#         ann_by_image[img_id] = []
#     ann_by_image[img_id].append(ann)

# print(f"📊 找到 {len(image_map)} 张图像的标注，开始绘制...")

# # 加载字体（如果没有就用默认）
# try:
#     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
# except:
#     font = ImageFont.load_default()

# for img_id, anns in tqdm(ann_by_image.items(), desc="Drawing boxes"):
#     img_info = image_map[img_id]
#     img_name = img_info["file_name"]
#     img_path = os.path.join(IMAGE_DIR, img_name)
    
#     if not os.path.exists(img_path):
#         print(f"⚠️ 图片不存在: {img_path}")
#         continue

#     image = Image.open(img_path).convert("RGB")
#     draw = ImageDraw.Draw(image)

#     for ann in anns:
#         bbox = ann["bbox"]          # [x, y, w, h]
#         cat_id = ann["category_id"]
#         score = ann.get("score", 0.95)
#         print(f"DEBUG [{img_name}]: bbox={bbox}, ImageSize=({image.width}, {image.height})")
#         cat_name = next((cat["name"] for cat in coco["categories"] if cat["id"] == cat_id), "unknown")
        
#         x, y, w, h = bbox
#         x1, y1 = int(x), int(y)
#         x2, y2 = int(x + w), int(y + h)
        
#         color = COLORS.get(cat_name, "#FFFFFF")
        
#         # 画矩形框
#         draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        
#         # 准备标签文字
#         label = f"{cat_name} {score:.2f}"
        
#         # 新版 Pillow 使用 textbbox 计算文字大小
#         bbox_text = draw.textbbox((0, 0), label, font=font)
#         tw = bbox_text[2] - bbox_text[0]
#         th = bbox_text[3] - bbox_text[1]
        
#         # 文字背景（让标签更清晰）
#         draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        
#         # 画文字
#         draw.text((x1 + 4, y1 - th - 4), label, fill="black", font=font)

#     # 保存可视化结果
#     save_path = os.path.join(VIS_DIR, img_name)
#     image.save(save_path)

# print(f"\n🎉 可视化完成！共处理 {len(ann_by_image)} 张图像")
# print(f"📁 查看路径: {VIS_DIR}")

import json
import cv2
import os
from matplotlib import pyplot as plt

# ================= 配置 =================
# JSON_PATH = "/root/autodl-tmp/bi/all_step4wE.json" # 你生成的 JSON
# IMAGE_DIR = "/root/autodl-tmp/bi/data/dataset/"             # 对应图片目录
# SAVE_DIR = "/root/autodl-tmp/bi/debug_visual/"               # 结果保存目录
JSON_PATH = "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json" # 你生成的 JSON
# JSON_PATH = "/root/autodl-tmp/bi/ROUD5_blur_vlm2.json" # 你生成的 JSON
IMAGE_DIR = "/root/autodl-tmp/bi/DGUNet/results/enhanced_blur"             # 对应图片目录
SAVE_DIR = "/root/autodl-tmp/bi/ROUD_visual/blur/dgunet"               # 结果保存目录
# JSON_PATH = "/root/autodl-tmp/bi/DUO_step4.json" # 你生成的 JSON
# IMAGE_DIR = "/root/autodl-tmp/bi/DUO"             # 对应图片目录
# SAVE_DIR = "/root/autodl-tmp/bi/DUO_visual/raw"               # 结果保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 颜色映射 (B, G, R)
COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), 
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128)
]

def draw_coco_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 建立类ID到名称的映射
    id_to_cat = {c['id']: c['name'] for c in data['categories']}
    
    # 建立图片ID到信息的映射
    images = {img['id']: img for img in data['images']}
    
    # 按 image_id 分组标注，方便一张张画
    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in data['annotations']:
        img_anns[ann['image_id']].append(ann)
    i = 0
    for img_id, anns in img_anns.items():
        # if i>50:break
        img_info = images[img_id]
        img_name = img_info['file_name']
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"找不到图片: {img_path}")
            continue

        image = cv2.imread(img_path)
        
        # --- 关键：去重后再画，防止复读机标注堆叠导致颜色变黑 ---
        unique_boxes = set()
        
        for ann in anns:
            cat_name = id_to_cat[ann['category_id']]
            bbox = ann['bbox'] # [x, y, w, h]
            
            # 使用元组记录坐标用于去重
            box_tuple = tuple(bbox)
            if box_tuple in unique_boxes: continue
            unique_boxes.add(box_tuple)

            x, y, w, h = [int(v) for v in bbox]
            color = COLORS[ann['category_id'] % len(COLORS)]

            # 画框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 写标签
            label = f"{cat_name}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存图片
        save_path = os.path.join(SAVE_DIR, f"res_{img_name}")
        cv2.imwrite(save_path, image)
        i+=1
        print(f"已保存可视化结果: {save_path}")

if __name__ == "__main__":
    draw_coco_results(JSON_PATH)
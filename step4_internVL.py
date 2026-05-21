import torch
import os
import json
import re
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from modelscope import snapshot_download

# ==================== 配置区 ====================
MODEL_PATH = "OpenGVLab/InternVL2_5-26B" 
BASE_DIR = "/root/autodl-tmp/bi/ROUD"
DIAGNOSIS_JSON = "all_step2_fixed.json" 
SUBSETS = {
    "blur": "instances_blur.json",
    "color": "instances_color.json",
    "turbid": "instances_turbid.json"
}
TARGET_CATEGORIES = [
    {"id": 1, "name": "holothurian"}, {"id": 2, "name": "echinus"},
    {"id": 3, "name": "scallop"}, {"id": 4, "name": "starfish"},
    {"id": 5, "name": "fish"}, {"id": 6, "name": "corals"},
    {"id": 7, "name": "diver"}, {"id": 8, "name": "cuttlefish"},
    {"id": 9, "name": "turtle"}, {"id": 10, "name": "jellyfish"}
]

# ==================== InternVL 图像预处理 ====================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def parse_to_coco_bbox(response, w, h):
    """
    解析 InternVL 输出的文本，提取类别和坐标，并转换为 COCO 格式的 bbox [x, y, width, height]。
    预期输入格式: <ref>category name</ref><box>[[ymin, xmin, ymax, xmax]]</box>
    """
    preds = []
    # 使用正则表达式匹配目标及其对应的框
    pattern = r'<ref>\s*(.*?)\s*</ref>\s*<box>\s*\[\[(.*?)\]\]\s*</box>'
    matches = re.findall(pattern, response)
    
    for cat_name, box_str in matches:
        try:
            # 提取坐标: ymin, xmin, ymax, xmax (0-1000的归一化值)
            coords = [float(x.strip()) for x in box_str.split(',')]
            if len(coords) != 4:
                continue
                
            ymin, xmin, ymax, xmax = coords
            
            # 将 0-1000 的坐标反归一化到原图的真实像素尺寸
            xmin_abs = (xmin / 1000.0) * w
            ymin_abs = (ymin / 1000.0) * h
            xmax_abs = (xmax / 1000.0) * w
            ymax_abs = (ymax / 1000.0) * h
            
            # 限制坐标不超出图像边界 (防越界保护)
            xmin_abs = max(0, min(xmin_abs, w))
            ymin_abs = max(0, min(ymin_abs, h))
            xmax_abs = max(0, min(xmax_abs, w))
            ymax_abs = max(0, min(ymax_abs, h))
            
            # 计算宽度和高度
            bbox_w = xmax_abs - xmin_abs
            bbox_h = ymax_abs - ymin_abs
            
            # 过滤掉面积无效的“幽灵框”
            if bbox_w <= 0 or bbox_h <= 0:
                continue
                
            preds.append({
                # 转为小写以匹配你的 TARGET_CATEGORIES 字典
                'category_name': cat_name.lower().strip(), 
                # 保留两位小数，减小 json 文件体积
                'bbox': [round(xmin_abs, 2), round(ymin_abs, 2), round(bbox_w, 2), round(bbox_h, 2)]
            })
            
        except Exception as e:
            # 偶尔模型可能输出乱码，跳过即可，不影响整个循环
            # print(f"⚠️ 坐标解析错误: {box_str}") 
            continue
            
    return preds
    
def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_for_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ==================== 核心逻辑 ====================
def load_diagnosis_map(json_path):
    print(f"📂 正在加载物理诊断数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    diag_map = {}
    for item in data:
        fname = os.path.basename(item['image_path'])
        eval_res = item.get('hybrid_evaluation', {})
        scores = eval_res.get('severity_scores', {})
        
        diag_map[fname] = {
            "L": scores.get('turbidity_score', 5.0),
            "C": scores.get('color_score', 5.0),
            "B": scores.get('brightness_score', 5.0)
        }
    return diag_map

def load_model():
    model_dir = '/root/autodl-tmp/models/OpenGVLab/InternVL2_5-26B'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print("🚀 正在针对 5090 环境加载模型...")
    model = AutoModel.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" 
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    return model, tokenizer
    
def run_prediction_with_physics():
    model, tokenizer = load_model()
    diag_map = load_diagnosis_map(DIAGNOSIS_JSON) 
    cat_to_id = {cat['name']: cat['id'] for cat in TARGET_CATEGORIES}
    
    for folder, json_name in SUBSETS.items():
        img_dir = os.path.join(BASE_DIR, folder)
        output_coco = {"images": [], "annotations": [], "categories": TARGET_CATEGORIES}
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])[:100]
        
        ann_id_counter = 1
        for idx, img_name in enumerate(tqdm(img_files, desc=f"Processing {folder}")):
            img_path = os.path.join(img_dir, img_name)
            
            # --- 修复 1: 使用官方函数处理图片，并在这里控制 max_num ---
            pixel_values = load_image_for_internvl(img_path, max_num=12).to(torch.bfloat16).cuda()
            
            # 获取原始图像尺寸（供后面的 parse_to_coco_bbox 使用）
            with Image.open(img_path) as tmp_img:
                w, h = tmp_img.size
            
            # --- 核心改进：注入物理先验 ---
            p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})
            
            physics_hint = (
                f"Underwater Environment Analysis: Turbidity is {p['L']}/10, Color Cast is {p['C']}/10, "
                f"and Brightness is {p['B']}/10. Please account for these degradations. "
                "Search specifically for targets that might be camouflaged or blurred."
            )
            
            prompt = (
                f"{physics_hint}\n"
                f"Task: Detect and provide bounding boxes for {', '.join(cat_to_id.keys())}.\n"
                f"Output format: <ref>category name</ref><box>[[ymin, xmin, ymax, xmax]]</box>"
            )

            # --- 修复 2: 传入 pixel_values，移除报错的 max_num ---
            with torch.no_grad():
                response = model.chat(tokenizer, pixel_values, prompt, dict(max_new_tokens=512, do_sample=False))
            
            # 假设你的环境中已经定义了 parse_to_coco_bbox 函数
            preds = parse_to_coco_bbox(response, w, h)
            
            # --- 修复 3: 定义 img_id 以防止 COCO 保存时报错 ---
            img_id = idx + 1 
            
            for pred in preds:
                cat_id = cat_to_id.get(pred['category_name'])
                if cat_id:
                    output_coco["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": pred['bbox'],
                        "area": round(pred['bbox'][2] * pred['bbox'][3], 2),
                        "iscrowd": 0,
                        "score": 0.95 
                    })
                    ann_id_counter += 1

        save_path = f"phys_aware_{json_name}"
        with open(save_path, "w") as f: 
            json.dump(output_coco, f, indent=2)

if __name__ == "__main__":
    run_prediction_with_physics()
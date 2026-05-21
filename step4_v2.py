import torch
import os
import json
import re
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# ==================== 配置区 ====================
MODEL_DIR = '/root/autodl-tmp/models/OpenGVLab/InternVL2_5-26B'
# BASE_DIR = "/root/autodl-tmp/bi/ROUD"
BASE_DIR = "/root/autodl-tmp/bi/data/dataset/"
DIAGNOSIS_JSON = "/root/autodl-tmp/bi/all_step2_fixed.json" 
OUTPUT_DIR = "/root/autodl-tmp/bi"

SUBSETS = {
    "blur": "instances_blur.json",
    "color": "instances_color.json",
    "light": "instances_light.json"
}

TARGET_CATEGORIES = [
    {"id": 1, "name": "holothurian"}, {"id": 2, "name": "echinus"},
    {"id": 3, "name": "scallop"}, {"id": 4, "name": "starfish"},
    {"id": 5, "name": "fish"}, {"id": 6, "name": "corals"},
    {"id": 7, "name": "diver"}, {"id": 8, "name": "cuttlefish"},
    {"id": 9, "name": "turtle"}, {"id": 10, "name": "jellyfish"}
]

# ==================== InternVL 图像预处理模块 ====================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

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

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_for_internvl(image_path, input_size=448, max_num=6):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# ==================== 后处理与工具函数 ====================
def parse_to_coco_bbox(response: str, w: int, h: int):
    """支持两种输出格式：带<ref><box>标签的 和 简洁的 category[[ymin,xmin,ymax,xmax]]"""
    preds = []
    
    # 格式1：官方标准 <ref>xxx</ref><box>[[...]]</box>
    pattern1 = r'<ref>\s*(.*?)\s*</ref>\s*<box>\s*\[\[(.*?)\]\]\s*</box>'
    for cat_name, box_str in re.findall(pattern1, response):
        preds.append((cat_name.strip(), box_str))
    
    # 格式2：模型现在输出的简洁版（你当前看到的）
    # 支持多行，每行可能是 "holothurian[[130, 675, 345, 800]]"
    pattern2 = r'(\w+)\s*\[\[([\d\s.,-]+)\]\]'
    for cat_name, box_str in re.findall(pattern2, response):
        preds.append((cat_name.strip(), box_str))
    
    parsed = []
    for cat_name, box_str in preds:
        try:
            # 清理并转 float
            coords = [float(x.strip()) for x in re.split(r'[, ]+', box_str) if x.strip()]
            if len(coords) != 4:
                continue
            ymin, xmin, ymax, xmax = coords
            
            # 反归一化（假设是 0~1000 归一化坐标）
            x_abs = (xmin / 1000.0) * w
            y_abs = (ymin / 1000.0) * h
            w_abs = ((xmax - xmin) / 1000.0) * w
            h_abs = ((ymax - ymin) / 1000.0) * h
            
            parsed.append({
                'category_name': cat_name.lower().strip(),
                'bbox': [round(max(0, x_abs), 2), round(max(0, y_abs), 2),
                         round(max(0, w_abs), 2), round(max(0, h_abs), 2)]
            })
        except:
            continue
    
    return parsed

def load_diagnosis_map(json_path):
    print(f"📂 正在加载物理诊断数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    diag_map = {}
    for item in data:
        fname = os.path.basename(item['image_path'])
        scores = item.get('hybrid_evaluation', {}).get('severity_scores', {})
        diag_map[fname] = {"L": scores.get('turbidity_score', 5.0), "C": scores.get('color_score', 5.0), "B": scores.get('brightness_score', 5.0)}
    return diag_map

def load_model():
    print("🚀 正在针对 5090 加载 InternVL2.5-26B (官方 8-bit 量化)...")
    
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,           # ← 改成 8-bit（官方唯一推荐的量化方式）
        low_cpu_mem_usage=True,
        use_flash_attn=True,         # ← 必须开启（你之前没装会自动 fallback 到 eager）
        trust_remote_code=True
    ).eval()                         # ← 官方写法，不加 device_map="auto"

    # 保险起见，强制视觉塔保持 bf16（防止任何意外）
    if hasattr(model, 'vision_model'):
        model.vision_model.to(torch.bfloat16)
    if hasattr(model, 'mlp1'):
        model.mlp1.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    print("✅ 模型加载完成（8-bit + flash-attn 已启用）")
    return model, tokenizer
# ==================== 主循环 ====================
def run_prediction_with_physics():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
        
    model, tokenizer = load_model()
    diag_map = load_diagnosis_map(DIAGNOSIS_JSON)
    cat_to_id = {cat['name'].lower().strip(): cat['id'] for cat in TARGET_CATEGORIES}

    # for folder, json_name in SUBSETS.items():
    img_dir = BASE_DIR
    # if not os.path.exists(img_dir):
    #     print(f"⚠️ 跳过缺失目录: {img_dir}")
    #     continue

    output_coco = {"images": [], "annotations": [], "categories": TARGET_CATEGORIES}
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
    ann_id_counter = 1
    for idx, img_name in enumerate(tqdm(img_files)):
        img_path = os.path.join(img_dir, img_name)
        img_id = idx + 1
        
        # --- 1. 图像处理与变量定义 ---
        try:
            # 修复 NameError：在这里显式定义并赋值 pixel_values
            pixel_values = load_image_for_internvl(img_path, max_num=6).to(torch.bfloat16).cuda()
            
            with Image.open(img_path) as tmp_img:
                w, h = tmp_img.size
            
            # 写入图像信息到 COCO 结构
            output_coco["images"].append({
                "id": img_id, 
                "file_name": img_name, 
                "width": w, 
                "height": h
            })
        except Exception as e:
            print(f"❌ 无法加载图像 {img_name}: {e}")
            continue

        # --- 2. 物理先验注入与指令强化 ---
        # 获取 L-C-B 评分（用于论文中的 Phys-Aware 实验）
        # p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})
        
        # # 关键修改：必须以 <image>\n 开头，否则模型会返回 "I'm unable to analyze images"
        # prompt = (
        #     "<image>\n"
        #     f"Please detect all instances of holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, and jellyfish. "
        #     f"Underwater Environment Info: Turbidity {p.get('L', 5.0)}, Color Cast {p.get('C', 5.0)}, Brightness {p.get('B', 5.0)}. "
        #     "Output: <ref>category name</ref><box>[[ymin, xmin, ymax, xmax]]</box>"
        # )
                    # --- 物理先验 + 超级严格格式控制 ---
        p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})
        
        prompt = (
            "<image>\n"
            "You are an expert underwater marine life detector. "
            "Your task is to detect ALL instances of the following 10 categories: "
            "holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.\n\n"
            
            "Environment Info: Turbidity {:.1f}, Color Cast {:.1f}, Brightness {:.1f}.\n\n"
            
            "STRICT OUTPUT RULES (must follow exactly):\n"
            "1. Output ONLY the detection lines, nothing else.\n"
            "2. NEVER output full-image boxes like [[0,0,1000,1000]].\n"
            "3. NEVER use *, **, explanations, descriptions, or refusal words.\n"
            "4. Use exactly this format for every object (one line per object):\n"
            "   category[[ymin, xmin, ymax, xmax]]\n\n"
            
            "Examples:\n"
            "holothurian[[130, 675, 345, 800]]\n"
            "echinus[[570, 650, 610, 700]]\n"
            "starfish[[200, 300, 400, 500]]\n\n"
            
            "Now detect all objects in the image and output ONLY in the format above.\n"
            "Begin output now:"
        ).format(p['L'], p['C'], p['B'])
        # --- 3. 推理与解析 ---
        try:
            with torch.no_grad():
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config=dict(max_new_tokens=512, do_sample=False),
                    history=None  # 重置历史，防止模型复读之前的拒答
                )
            # 打印输出以便在 AutoDL 后台监控效果
            if idx % 10 == 0: 
                print(f"\nDEBUG - {img_name}: {response[:100]}...")
        except Exception as e:
            print(f"⚠️ 推理异常 {img_name}: {e}")
            continue

        # --- 4. 解析结果并写入 COCO ---
        preds = parse_to_coco_bbox(response, w, h)
        
        for pred in preds:
            cat_name = pred['category_name'].lower().strip()
            cat_id = cat_to_id.get(cat_name)
            
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

    # 保存该子集的 JSON 结果
    save_path = os.path.join(OUTPUT_DIR, f"all_step4")
    with open(save_path, "w") as f: 
        json.dump(output_coco, f, indent=2)
        
    print(f"\n✅ 子集 {folder} 处理完成！")
    print(f"   - 图片数量: {len(output_coco['images'])}")
    print(f"   - 标注数量: {len(output_coco['annotations'])}")
    print(f"   - 结果路径: {save_path}")
if __name__ == "__main__":
    run_prediction_with_physics()
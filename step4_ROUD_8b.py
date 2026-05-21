import torch
import os
import json
import re
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import gc

# ==================== 内存优化 ====================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== 配置区 ====================
MODEL_DIR = '/root/autodl-tmp/models/OpenGVLab/InternVL2_5-26B'
BASE_DIR = "/root/autodl-tmp/bi/ROUD"
DIAGNOSIS_JSON = "all_step2_fixed.json"
GT_DIR = "/root/autodl-tmp/bi/results"

SUBSETS = {
    "blur": "instances_blur.json",
    # "color": "instances_color.json",
    # "light": "instances_light.json"
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
    preds = []
    pattern = r'([a-zA-Z\s\-_]+)\[\[\s*([\d\s.,]+)\s*\]\]'
    matches = re.findall(pattern, response)
   
    for cat_name, box_str in matches:
        try:
            coords = [float(x.strip()) for x in re.split(r'[,\s]+', box_str.strip()) if x.strip()]
            if len(coords) != 4:
                continue
            ymin, xmin, ymax, xmax = coords
           
            real_xmin = (xmin / 1000.0) * w
            real_ymin = (ymin / 1000.0) * h
            real_xmax = (xmax / 1000.0) * w
            real_ymax = (ymax / 1000.0) * h
           
            bbox_w = real_xmax - real_xmin
            bbox_h = real_ymax - real_ymin
           
            real_xmin = max(0, min(real_xmin, w))
            real_ymin = max(0, min(real_ymin, h))
            bbox_w = max(0, min(bbox_w, w - real_xmin))
            bbox_h = max(0, min(bbox_h, h - real_ymin))
           
            if bbox_w <= 1 or bbox_h <= 1:
                continue
               
            preds.append({
                'category_name': cat_name.lower().strip(),
                'bbox': [round(real_xmin, 2), round(real_ymin, 2), round(bbox_w, 2), round(bbox_h, 2)]
            })
        except Exception:
            continue
    return preds

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=False):  # 已改小为4
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
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

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

def load_image_for_internvl(image_file, input_size=448, max_num=4):   # 默认改成4，大幅省显存
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# ==================== 加载函数 ====================
def load_diagnosis_map(json_path):
    print(f"📂 正在加载物理诊断数据: {json_path}")
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到 {json_path}，使用默认评分。")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    diag_map = {}
    for item in data:
        fname = os.path.basename(item['image_path'])
        scores = item.get('hybrid_evaluation', {}).get('severity_scores', {})
        diag_map[fname] = {
            "L": scores.get('turbidity_score', 5.0),
            "C": scores.get('color_score', 5.0),
            "B": scores.get('brightness_score', 5.0)
        }
    return diag_map

def load_model():
    print(f"🚀 正在加载模型: {MODEL_DIR} (非量化 bf16 模式)")
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    print("模型成员:", [n for n, _ in model.named_children()])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    return model, tokenizer

# ==================== 主函数（支持断点续跑 + 边跑边保存） ====================
def run_prediction_with_physics():
    model, tokenizer = load_model()
    diag_map = load_diagnosis_map(DIAGNOSIS_JSON)
    cat_to_id = {cat['name'].lower(): cat['id'] for cat in TARGET_CATEGORIES}
   
    for folder, json_name in SUBSETS.items():
        img_dir = os.path.join(BASE_DIR, folder)
        if not os.path.exists(img_dir):
            print(f"⚠️ 跳过不存在的目录: {img_dir}")
            continue
           
        save_path = f"step4_ROUD8b_{json_name}"
        
        # 加载已保存的结果（支持断点续跑）
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                output_coco = json.load(f)
            print(f"✅ 发现已保存文件 {save_path}，已处理 {len(output_coco['images'])} 张图片，继续运行...")
        else:
            output_coco = {"images": [], "annotations": [], "categories": TARGET_CATEGORIES}
        
        # 已处理的 image_id 集合，用于跳过
        processed_ids = {img["id"] for img in output_coco["images"]}
        
        gt_path = os.path.join(GT_DIR, json_name)
        fname_to_id = {}
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            fname_to_id = {img['file_name']: img['id'] for img in gt_data['images']}
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
       
        ann_id_counter = len(output_coco["annotations"]) + 1   # 从已保存的继续计数
        
        for idx, img_name in enumerate(tqdm(img_files, desc=f"Processing {folder}")):
            img_id = fname_to_id.get(img_name, idx + 1)
            
            # 如果已经处理过，跳过
            if img_id in processed_ids:
                continue
                
            img_path = os.path.join(img_dir, img_name)
           
            try:
                # 加载图像
                pixel_values = load_image_for_internvl(img_path, max_num=4).to(torch.bfloat16).cuda()
                with Image.open(img_path) as tmp_img:
                    w, h = tmp_img.size
                
                # 构造 prompt
                p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})
                prompt = (
                    "<image>\n"
                    "You are an expert underwater marine life detector specialized in challenging underwater conditions.\n\n"
                    "Detect ALL visible instances of the following categories ONLY:\n"
                    "holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.\n\n"
                    "Current Environment: Turbidity {:.1f}/10, Color Cast {:.1f}/10, Brightness {:.1f}/10.\n"
                    "STRICT OUTPUT RULES:\n"
                    "- Output ONE line per detected object.\n"
                    "- Format exactly: category[[ymin, xmin, ymax, xmax]]\n"
                    "- Coordinates normalized 0-1000.\n"
                    "- Use ONLY the exact category names (lowercase).\n"
                    "- If no objects are clearly visible, output nothing.\n\n"
                    "Examples:\n"
                    "holothurian[[130, 675, 345, 800]]\n"
                    "starfish[[200, 300, 400, 500]]\n\n"
                    "Begin:"
                ).format(p['L'], p['C'], p['B'])
                
                # 推理
                with torch.no_grad():
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        prompt,
                        generation_config=dict(max_new_tokens=512, do_sample=False),  # 降低生成长度
                        history=None,
                    )
                
                # 解析并添加结果
                preds = parse_to_coco_bbox(response, w, h)
                
                output_coco["images"].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": w,
                    "height": h
                })
                
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
                
                # === 关键：每张图处理完立即保存 ===
                with open(save_path, "w") as f:
                    json.dump(output_coco, f, indent=2)
                
                # 清理显存
                del pixel_values
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"\n❌ Error on {img_name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue   # 继续处理下一张
        
        print(f"\n✅ {folder} 处理完成！结果已保存至: {save_path}\n")

if __name__ == "__main__":
    run_prediction_with_physics()
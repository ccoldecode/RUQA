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
BASE_DIR = "/root/autodl-tmp/bi/ROUD"
DIAGNOSIS_JSON = "all_step2_fixed.json" 
# 结果保存路径（确保与真值表路径区分开）
GT_DIR = "/root/autodl-tmp/bi/results" 
SUBSETS = {
    "blur": "instances_blur.json"#,
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
    """
    适配提示词格式: category[[ymin, xmin, ymax, xmax]]
    转换逻辑：归一化坐标 -> 像素坐标 -> [x, y, w, h]
    """
    preds = []
    # 匹配模式：类别名[[坐标数字]]
    pattern = r'([a-zA-Z\s\-_]+)\[\[\s*([\d\s.,]+)\s*\]\]'
    matches = re.findall(pattern, response)
    
    for cat_name, box_str in matches:
        try:
            # 分离坐标数字
            coords = [float(x.strip()) for x in re.split(r'[,\s]+', box_str.strip()) if x.strip()]
            if len(coords) != 4:
                continue
                
            ymin, xmin, ymax, xmax = coords
            
            # 1. 反归一化 (InternVL 使用 0-1000 体系)
            real_xmin = (xmin / 1000.0) * w
            real_ymin = (ymin / 1000.0) * h
            real_xmax = (xmax / 1000.0) * w
            real_ymax = (ymax / 1000.0) * h
            
            # 2. 转换为 COCO 格式 [x_min, y_min, width, height]
            bbox_w = real_xmax - real_xmin
            bbox_h = real_ymax - real_ymin
            
            # 3. 边界限制与非法过滤
            real_xmin = max(0, min(real_xmin, w))
            real_ymin = max(0, min(real_ymin, h))
            bbox_w = max(0, min(bbox_w, w - real_xmin))
            bbox_h = max(0, min(bbox_h, h - real_ymin))
            
            if bbox_w <= 1 or bbox_h <= 1: # 过滤面积极小值
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

def load_image_for_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# ==================== 核心加载逻辑 ====================
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

# def load_model():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )

#     print(f"🚀 正在加载模型: {MODEL_DIR}")
#     # 修改 load_model 函数中的 model 加载部分
#     model = AutoModel.from_pretrained(
#         MODEL_DIR,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         local_files_only=True 
#     ).eval()
    
#     # 检查一下：如果 model 对象没有 'visual_encoder' 属性，说明加载失败
#     if not hasattr(model, 'visual_encoder'):
#         print("❌ 警告：视觉编码器未加载！请检查模型文件是否完整（需包含 pytorch_model.bin.index.json 等）")

#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_DIR, 
#         trust_remote_code=True, 
#         use_fast=False,
#         local_files_only=True
#     )
#     return model, tokenizer

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    print(f"🚀 正在加载模型: {MODEL_DIR}")
    
    # 1. 先加载配置，看看能不能读到视觉部分
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # 2. 显式指定加载
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        config=config, # 传入显式配置
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    # 3. 验证视觉编码器（InternVL2.5 内部通常叫 vision_model 或 visual_encoder）
    # 尝试打印模型结构的一部分来确认
    print("模型成员:", [n for n, _ in model.named_children()])
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, 
        trust_remote_code=True, 
        use_fast=False
    )
    return model, tokenizer
def run_prediction_with_physics():
    model, tokenizer = load_model()
    diag_map = load_diagnosis_map(DIAGNOSIS_JSON) 
    cat_to_id = {cat['name'].lower(): cat['id'] for cat in TARGET_CATEGORIES}
    
    for folder, json_name in SUBSETS.items():
        img_dir = os.path.join(BASE_DIR, folder)
        if not os.path.exists(img_dir):
            print(f"⚠️ 跳过不存在的目录: {img_dir}")
            continue
            
        output_coco = {"images": [], "annotations": [], "categories": TARGET_CATEGORIES}
        
        # --- 核心：官方 ID 对齐逻辑 ---
        gt_path = os.path.join(GT_DIR, json_name)
        fname_to_id = {}
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            fname_to_id = {img['file_name']: img['id'] for img in gt_data['images']}
            print(f"✅ 成功加载真值表 {json_name}，已开启官方 ID 自动对齐。")
        else:
            print(f"⚠️ 未找到真值表 {gt_path}，评测 mAP 可能会失败！")
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        ann_id_counter = 1
        for idx, img_name in enumerate(tqdm(img_files, desc=f"Processing {folder}")):
            img_path = os.path.join(img_dir, img_name)
            
            try:
                pixel_values = load_image_for_internvl(img_path, max_num=12).to(torch.bfloat16).cuda()
                with Image.open(img_path) as tmp_img:
                    w, h = tmp_img.size
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
                
            # 获取官方 Image ID
            img_id = fname_to_id.get(img_name, idx + 1) 
            
            output_coco["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })
            
            p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})

            prompt = (
                "<image>\n"
                "You are an expert underwater marine life detector. "
                "Your task is to detect ALL instances of: "
                "holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.\n\n"
                
                "Current Environment Info: Turbidity {:.1f}/10, Color Cast {:.1f}/10, Brightness {:.1f}/10. "
                "Note: Even if the image is blurry due to these conditions, use your visual prior to estimate the bounding boxes.\n\n"
                
                "OUTPUT FORMAT (One line per object):\n"
                "category[[ymin, xmin, ymax, xmax]]\n\n"
                
                "Examples for coordinate system (0-1000):\n"
                "holothurian[[130, 675, 345, 800]]\n"
                "starfish[[200, 300, 400, 500]]\n\n"
                
                "Detect all objects and output ONLY the category and coordinates. Begin:"
            ).format(p['L'], p['C'], p['B'])
            # p = diag_map.get(img_name, {"L": 5.0, "C": 5.0, "B": 5.0})

            # # 核心改动：在结尾增加一个明确的起始符，并放宽“看不清”时的限制
            # prompt = (
            #     "<image>\n"
            #     "You are an expert underwater marine life detector. "
            #     "Task: Detect ALL instances of: holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.\n\n"
                
            #     "Environment: Turbidity {:.1f}/10, Color Cast {:.1f}/10, Brightness {:.1f}/10.\n"
            #     "Visual Guide: The water is murky, but objects are present. Use your underwater vision to find them.\n\n"
                
            #     "STRICT FORMAT (One line per object, coordinates 0-1000):\n"
            #     "category[[ymin, xmin, ymax, xmax]]\n\n"
                
            #     "Example:\n"
            #     "starfish[[200, 300, 400, 500]]\n\n"
                
            #     "Now, analyze the image carefully. If no objects are found, output 'none'. "
            #     "Otherwise, output the detections. Begin:"
            # ).format(p['L'], p['C'], p['B'])
            try:
                with torch.no_grad():
                    print(f"DEBUG - Image Tensor Shape: {pixel_values.shape}")
                    response = model.chat(
                        tokenizer, 
                        pixel_values, 
                        prompt, 
                        generation_config=dict(max_new_tokens=1024, do_sample=False),
                        history=None,
                        # model_name='internvl2_5' # 显式告诉它使用 InternVL2.5 的模板
                    )
                    # 调试：看看模型到底吐了什么字符
                    print(f"--- 原图: {img_name} ---")
                    print(f"模型原始输出: {response}")
            except Exception as e:
                print(f"Inference error on {img_name}: {e}")
                continue
            
            # 解析输出并保存
            preds = parse_to_coco_bbox(response, w, h)
            
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

        # 保存结果
        save_path = f"step4_ROUD_{json_name}"
        with open(save_path, "w") as f: 
            json.dump(output_coco, f, indent=2)
        print(f"\n✅ {folder} 完成！结果已保存至: {save_path}\n")

if __name__ == "__main__":
    run_prediction_with_physics()
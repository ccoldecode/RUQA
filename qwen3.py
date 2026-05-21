import torch
import os
import json
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

# ================= 1. 配置 =================
MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b"
JSON_INFO_PATH = "/root/autodl-tmp/bi/DUO_step2_fixed.json"
RAW_IMAGE_DIR = "/root/autodl-tmp/bi/DUO"
ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DTIUIE_DUO"
# ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DGUNet/results/enhanced_light"
# MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b"
# JSON_INFO_PATH = "/root/autodl-tmp/bi/all_step2_fixed.json"
# RAW_IMAGE_DIR = "/root/autodl-tmp/bi/dataset"
# ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DTIUIE_output2"

USE_ENHANCED = False  # ← 改成 True 就是用增强图

OUTPUT_JSON = "/root/autodl-tmp/bi/DUO_step4.json"

# CATEGORY_MAP = {
#     "holothurian": 1, "echinus": 2, "scallop": 3, "starfish": 4, "fish": 5,
#     "corals": 6, "diver": 7, "squid": 8, "turtle": 9, "jellyfish": 10
# }
#DUO
CATEGORY_MAP = {
    "holothurian": 1, "echinus": 2, "scallop": 3, "starfish": 4
}

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# ================= 2. 加载模型 =================
print(f"🚀 加载模型... 模式: {'增强图' if USE_ENHANCED else '原始图'}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    trust_remote_code=True
)

# ================= 3. Prompt =================
def build_prior_prompt(entry):
    reasoning = entry.get('hybrid_evaluation', {}).get('reasoning', "")
    if not reasoning:
        hybrid = entry.get('hybrid_evaluation', {})
        t_score = hybrid.get('turbidity_blur', {}).get('score', 1)
        c_type = hybrid.get('color_cast', {}).get('type', 'none')
        hints = []
        if t_score >= 6:
            hints.append("murky water")
        if c_type.lower() != "none":
            hints.append(f"{c_type} color cast")
        reasoning = ", ".join(hints) if hints else "clear water"
    
    cats = ", ".join(CATEGORY_MAP.keys())

    needs_enhancement = entry.get('hybrid_evaluation', {}).get('needs_enhancement', "")
#版本5
    prompt = f"""You are an expert underwater object detector with strict confidence calibration.

Visual Context: {reasoning}

Task: Detect all underwater objects. Categories: {cats}

For each detected object, you MUST output:
- "bbox_2d": [x1, y1, x2, y2] (tight bounding box in absolute pixel coordinates)
- "label": "category_name"
- "confidence": a float between 0.0 and 1.0 that accurately reflects your certainty:
    - 0.95~1.00 : Very high confidence (object is clear and obvious)
    - 0.80~0.94 : High confidence (object is visible but slightly unclear)
    - 0.60~0.79 : Medium confidence (object is blurry, small, or partially occluded)
    - 0.40~0.59 : Low confidence (guess based on shape/color)
    - < 0.40  : Very uncertain, but still output if you think it might be there

Output strictly in JSON format as a list, no extra text:
[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "label": "category_name",
    "confidence": 0.XX
  }}
]
"""
#版本4
    #[Condition]
    # This image has been enhanced for better visibility: {needs_enhancement}. 
    # (If true, leverage the restored details and colors to identify small or camouflaged objects.)    
#     prompt = f"""You are an expert underwater object detector.
# Visual Context: {reasoning}

# [Target Categories]
# {cats}

# [Condition]
# This image has been enhanced for better visibility: {needs_enhancement}. 
# (If true, leverage the restored details and colors to identify small or camouflaged objects.)
    
# [Task]
# Detect ALL objects from the target categories. Ensure every visible instance is captured.

# [Output Rules]
# - Use absolute pixel coordinates for "bbox_2d": [x1, y1, x2, y2].
# - Output **strictly** in the following JSON format (no extra text, no markdown):
# [
#   {{
#     "bbox_2d": [x1, y1, x2, y2],
#     "label": "category_name",
#     "confidence": 0.XX
#   }}
# ]
# """
    
#版本3
#     # 1. 定义提示词模板的不同部分
#     BASE_SYSTEM_PROMPT = """You are an expert underwater object detector.
# Visual Context: {reasoning}

# Task:
# 1. Detect **ALL** visible underwater objects. Do not miss any, even small or partially obscured ones.
# 2. Categories: {cats}
# """

# # 针对【已增强】图片的提示词片段
#     ENHANCED_FRAGMENT = """
# Background Note: 
# This image has been ENHANCED using DGUNet. While clearer, it may contain artificial textures or sharpening artifacts. 
# - Focus on distinguishing real marine life from enhancement-induced noise.
# - Restored colors may slightly deviate from natural appearances.
# """

# # 针对【原始/未增强】图片的提示词片段
#     RAW_FRAGMENT = """
# Special underwater challenges to consider:
# - Low visibility, color attenuation (red disappears first), haze/blur.
# - Objects may blend with background (coral, sand, rocks).
# - Strong light scattering and caustics.
# """

#     BASE_FOOTER = """
# Output **strictly** in the following JSON format:
# [
#   {{
#     "bbox_2d": [x1, y1, x2, y2],
#     "label": "category_name",
#     "confidence": 0.XX
#   }}
# ]

# Rules:
# - Every detected object must have a bbox that tightly fits the object.
# - If multiple objects overlap, still output separate entries.
# """

#     # 2. 动态逻辑判定
#     needs_enhancement = entry.get('hybrid_evaluation', {}).get('needs_enhancement', "")
    
#     # 兼容字符串 "true" 或 布尔值 True
#     if str(needs_enhancement).lower() == "true":
#         dynamic_context = ENHANCED_FRAGMENT
#     else:
#         dynamic_context = RAW_FRAGMENT
    
#     # 3. 最终拼接
#     prompt = f"{BASE_SYSTEM_PROMPT}\n{dynamic_context}\n{BASE_FOOTER}"

#版本2
#     prompt = f"""You are an expert underwater object detector.

# Visual Context: {reasoning}

# Task:
# 1. Detect **ALL** visible underwater objects. Do not miss any, even small or partially obscured ones.
# 2. Categories: {cats}
# 3. Special underwater challenges to consider:
#    - Low visibility, color attenuation (red disappears first), haze/blur
#    - Objects may blend with background (coral, sand, rocks)
#    - Strong light scattering and caustics

# Output **strictly** in the following JSON format (list of objects, no extra text, no markdown):
# [
#   {{
#     "bbox_2d": [x1, y1, x2, y2],   // absolute pixel coordinates, x1 < x2, y1 < y2
#     "label": "category_name",
#     "confidence": 0.XX            // optional but recommended, 0.0-1.0
#   }}
# ]

# Rules:
# - Every detected object must have a bbox that tightly fits the object.
# - If multiple objects overlap, still output separate entries.
# - Do not output anything outside the JSON list.
# """
    return prompt

# ================= 4. 主循环 =================
with open(JSON_INFO_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

test_data = data

coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": n} for n, i in CATEGORY_MAP.items()]
}

ann_id = 1
img_id = 1

for entry in tqdm(test_data, desc="Qwen Inference"):
    raw_path = entry.get('image_path', '')
    img_name = os.path.basename(raw_path)
    img_path = os.path.join(ENHANCED_IMAGE_DIR if USE_ENHANCED else RAW_IMAGE_DIR, img_name)
   
    if not os.path.exists(img_path):
        continue

    # 计算缩放系数
    img_obj = Image.open(img_path)
    o_w, o_h = img_obj.size
    r_h, r_w = smart_resize(o_h, o_w, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    s_w = o_w / r_w
    s_h = o_h / r_h

    # 推理
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{img_path}"},
        {"type": "text", "text": build_prior_prompt(entry)}
    ]}]
   
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
   
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.15
        )
        output_text = processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

    coco_output["images"].append({"id": img_id, "file_name": img_name, "width": o_w, "height": o_h})

    # 解析输出
    seen_boxes = set()
    # try:
    #     # clean_json = re.sub(r'```json\s*|\s*```', '', output_text).strip()
    #     match = re.search(r'\[.*\]', output_text, re.DOTALL)
    #     if match:
    #         clean_json = match.group(0)
    #     else:
    #         # 如果没搜到中括号，再尝试之前的清理逻辑作为兜底
    #         clean_json = re.sub(r'```json\s*|\s*```', '', output_text).strip()
    #     detections = json.loads(clean_json)
       
    #     for det in detections:
    #         label = det.get("label", "").lower().strip()
    #         bbox = det.get("bbox_2d", [])
           
    #         cid = CATEGORY_MAP.get(label, -1)
    #         if cid != -1 and len(bbox) == 4:
    #             x1, y1, x2, y2 = bbox
    #             rx = round(max(0, min(x1, x2) * s_w), 2)
    #             ry = round(max(0, min(y1, y2) * s_h), 2)
    #             rw = round(abs(x2 - x1) * s_w, 2)
    #             rh = round(abs(y2 - y1) * s_h, 2)
                
    #             box_key = (cid, round(rx), round(ry))
    #             if box_key not in seen_boxes:
    #                 seen_boxes.add(box_key)
    #                 if rw > 2 and rh > 2:
    #                     coco_output["annotations"].append({
    #                         "id": ann_id,
    #                         "image_id": img_id,
    #                         "category_id": cid,
    #                         "bbox": [rx, ry, rw, rh],
    #                         "score": 0.95,
    #                         "area": round(rw * rh, 2),
    #                         "iscrowd": 0
    #                     })
    #                     ann_id += 1
    # except Exception as e:
    #     truncated_output = output_text[:100].replace('\n', ' ')
    #     print(f"\n⚠️ [{img_name}] 解析失败: {e} | 输出片段: {truncated_output}...")
    try:
        # 1. 尝试提取中括号内的内容
        match = re.search(r'\[.*\]', output_text, re.DOTALL)
        content_to_parse = match.group(0) if match else output_text
        
        # 2. 清理 Markdown 标签
        content_to_parse = re.sub(r'```json\s*|\s*```', '', content_to_parse).strip()
        
        detections = []
        try:
            # 尝试标准解析
            detections = json.loads(content_to_parse)
        except json.JSONDecodeError:
            # 【核心修复逻辑】如果标准解析失败，说明 JSON 末尾截断了
            # 使用正则强行匹配每一个完整的 { "bbox_2d": ... } 对象
            # 这样即便最后几个字符丢了，前面的框也能保住
            print(f" ⚠️ [{img_name}] 检测到 JSON 截断，启动正则强行挽救框数据...")
            object_matches = re.finditer(r'\{[^{}]*\}', content_to_parse, re.DOTALL)
            for m in object_matches:
                try:
                    obj = json.loads(m.group(0))
                    detections.append(obj)
                except:
                    continue # 略过真正损坏的单个对象

        # --- 以下处理逻辑保持不变 ---
        for det in detections:
            label = det.get("label", "").lower().strip()
            bbox = det.get("bbox_2d", [])
            
            cid = CATEGORY_MAP.get(label, -1)
            if cid != -1 and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                rx = round(max(0, min(x1, x2) * s_w), 2)
                ry = round(max(0, min(y1, y2) * s_h), 2)
                rw = round(abs(x2 - x1) * s_w, 2)
                rh = round(abs(y2 - y1) * s_h, 2)
                
                box_key = (cid, round(rx), round(ry))
                if box_key not in seen_boxes:
                    seen_boxes.add(box_key)
                    if rw > 2 and rh > 2:
                        coco_output["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cid,
                            "bbox": [rx, ry, rw, rh],
                            "score": det.get("confidence", 0.75), # 优先使用模型返回的置信度
                            "area": round(rw * rh, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1
                        
    except Exception as e:
        truncated_output = output_text[:100].replace('\n', ' ')
        print(f"\n❌ [{img_name}] 彻底解析失败: {e} | 输出片段: {truncated_output}...")

    img_id += 1

# ================= 5. 保存 =================
with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
    json.dump(coco_output, f, indent=4, ensure_ascii=False)

print(f"\n✅ 处理完成！结果已存至: {OUTPUT_JSON}")
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
JSON_INFO_PATH = "/root/autodl-tmp/bi/ROUD_step2_fixed_color.json"
RAW_IMAGE_DIR = "/root/autodl-tmp/bi/ROUD/color"
ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DTIUIE_ROUD_color"
# MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b"
# JSON_INFO_PATH = "/root/autodl-tmp/bi/all_step2_fixed.json"
# RAW_IMAGE_DIR = "/root/autodl-tmp/bi/dataset"
# ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DTIUIE_output2"

USE_ENHANCED = False   # ← 改成 True 就是用增强图

OUTPUT_JSON = "/root/autodl-tmp/bi/ROUD_step4_color.json"

CATEGORY_MAP = {
    "sea cucumber": 1, "sea urchin": 2, "scallop": 3, "starfish": 4, "fish": 5,
    "corals": 6, "diver": 7, "squid": 8, "turtle": 9, "jellyfish": 10
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
    
    prompt = f"""Visual Context: {reasoning}
Task: Detect all underwater objects (categories: {cats}).
Output strictly in JSON format as a list:
[ {{ "bbox_2d": [x1, y1, x2, y2], "label": "category_name" }} ]
"""
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
    try:
        clean_json = re.sub(r'```json\s*|\s*```', '', output_text).strip()
        detections = json.loads(clean_json)
       
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
                            "score": 0.95,
                            "area": round(rw * rh, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1
    except Exception as e:
        print(f"\n⚠️ [{img_name}] 解析失败: {e}")

    img_id += 1

# ================= 5. 保存 =================
with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
    json.dump(coco_output, f, indent=4, ensure_ascii=False)

print(f"\n✅ 处理完成！结果已存至: {OUTPUT_JSON}")
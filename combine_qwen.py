import torch
import json
import os
import cv2
import re
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- 1. 参数与路径配置 ---
MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b" 
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
IMG_ROOT = "/root/autodl-tmp/bi/ROUD/blur"
OUTPUT_PATH = "/root/autodl-tmp/bi/ROUD5_blur_vlm4.json"

JSON_FILES = {
    "original": "/root/autodl-tmp/bi/ROUD5_step4_blur.json",
    "dtiuie":   "/root/autodl-tmp/bi/ROUD5_step4_blurwE.json",
    "dgunet":   "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json"
}

CATEGORY_MAP = {
    "holothurian": 1, "echinus": 2, "scallop": 3, "starfish": 4, "fish": 5,
    "corals": 6, "diver": 7, "squid": 8, "turtle": 9, "jellyfish": 10
}

# --- 2. 提前加载模型 ---
print("Loading Model & Processor...")
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

# --- 3. 辅助函数 ---
def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def xyxy_to_coco(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

def get_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def run_vlm_judge(model, processor, image_path, candidates):
    if not candidates: return []
    img = cv2.imread(image_path)
    if img is None: return []
    marked_img = img.copy()
    for cand in candidates:
        x1, y1, x2, y2 = map(int, cand['xyxy'])
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(marked_img, cand['cid'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    tmp_path = f"/tmp/judge_{os.path.basename(image_path)}" # 避免并发冲突
    cv2.imwrite(tmp_path, marked_img)

    cat_str = ", ".join(CATEGORY_MAP.keys())
    cand_info = "\n".join([f"- {c['cid']}: {c['label']}" for c in candidates])
    
    prompt = f"""You are an Expert Marine Biologist. Identify which boxes truly contain a marine organism. REJECT noise or artifacts.
[Categories] {cat_str}
[Candidates]
{cand_info}
Output JSON: [ {{"id": "A", "verdict": "KEEP"}}, ... ]"""

    messages = [{"role": "user", "content": [{"type": "image", "image": tmp_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    try:
        json_str = re.search(r'\[\s*\{.*\}\s*\]', output_text, re.DOTALL).group(0)
        verdicts = json.loads(json_str)
        keep_ids = [v['id'] for v in verdicts if v.get('verdict') == 'KEEP']
        return [c for c in candidates if c['cid'] in keep_ids]
    except:
        return []

# --- 4. 数据加载 (修正后的严谨版) ---
raw_data = {}
all_anns_by_img = defaultdict(list)

for name, path in JSON_FILES.items():
    print(f"Reading {name}...")
    with open(path, 'r') as f:
        raw_data[name] = json.load(f)

# 建立统一的 COCO 模板
base_coco_data = {
    "images": raw_data["original"].get("images", []),
    "categories": raw_data["original"].get("categories", []),
    "annotations": []
}

# 建立索引：ID -> 完整的 Image 对象
id_to_img = {img["id"]: img for img in base_coco_data["images"]}
ID_TO_NAME = {cat['id']: cat['name'] for cat in base_coco_data['categories']}

for name, js in raw_data.items():
    for ann in js.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in id_to_img:
            all_anns_by_img[img_id].append(ann)

# --- 5. 审判循环 ---
new_annotations = []
ann_id_counter = 1

print(f"Starting VLM Judge for {len(all_anns_by_img)} images...")
for img_id, anns in tqdm(all_anns_by_img.items()):
    img_info = id_to_img.get(img_id)
    if not img_info: continue
    
    img_name = os.path.basename(img_info['file_name'])
    img_path = os.path.join(IMG_ROOT, img_name)
    
    if not os.path.exists(img_path):
        print(f"Error: Missing image at {img_path}")
        continue

    # a. 空间归并
    candidates = []
    for ann in anns:
        box_xyxy = coco_to_xyxy(ann['bbox'])
        is_dup = False
        for cand in candidates:
            if get_iou(box_xyxy, cand['xyxy']) > 0.7:
                is_dup = True
                break
        if not is_dup:
            candidates.append({
                'xyxy': box_xyxy,
                'label': ID_TO_NAME.get(ann['category_id'], "unknown"),
                'category_id': ann['category_id'],
                'score': ann.get('score', 0.75)
            })
    
    for i, c in enumerate(candidates):
        c['cid'] = chr(65 + i) if i < 26 else f"Z{i}"

    # b. 运行裁判
    final_keep = run_vlm_judge(model, processor, img_path, candidates)

    # c. 转换
    # --- 修改循环中的转换部分 ---
# 版本4
    for k in candidates:
        # 1. 严格锁定原始 ID，不要让 VLM 碰它
        original_category_id = k['category_id'] 
        
        # 2. 只从 VLM 结果中提取 verdict
        is_kept = any(fk['cid'] == k['cid'] and fk.get('verdict') == 'KEEP' for fk in final_keep)
        
        # 3. 软加权逻辑（vlm2 成功的关键）
        final_score = k['score'] if is_kept else 0.5 # 建议试下 0.5 或 0.6
        
        # 4. 强制写回
        new_annotations.append({
            "image_id": img_id,
            "category_id": original_category_id, # 必须是最初读取 JSON 时的那个 ID
            "bbox": xyxy_to_coco(k['xyxy']),      # 必须是最初的坐标
            "score": round(final_score, 4),
            "area": k.get('area', 0),
            "iscrowd": 0
    })
#版本3
    # for k in candidates:
    #     is_kept = any(fk['cid'] == k['cid'] for fk in final_keep)
        
    #     # 假设 count 是这个框被多少个原始 JSON 包含（1~3之间）
    #     # 你可以在空间归并阶段记录这个 count
    #     count = k.get('consensus_count', 1) 
        
    #     if is_kept:
    #         # 表现优异：VLM 和 检测器 达成共识
    #         final_score = min(1.0, k['score'] * (1.0 + 0.05 * count))
    #     else:
    #         # 分歧处理：
    #         if count >= 2:
    #             # 多个模型都看到了，VLM 可能是看漏了，轻微降分
    #             final_score = k['score'] * 0.8 
    #         else:
    #             # 只有一个模型看到且 VLM 反对，大概率是噪声，大幅降分
    #             final_score = 0.35
#版本2
    # for k in candidates:
    #     # 查找 VLM 的判定
    #     is_kept = any(fk['cid'] == k['cid'] for fk in final_keep)
        
    #     orig_score = k['score']  # 此时可能是 0.9 或 0.95
        
    #     if is_kept:
    #         # VLM 认可：保持高分
    #         final_score = orig_score 
    #     else:
    #         # VLM 不认可：
    #         # 这里是关键！不要删掉，而是将它降级到“低置信度区间”
    #         # 比如降到 0.4，这样它不会干扰前排的 Precision，但能保住 Recall
    #         final_score = 0.4  
    
        # coco_box = xyxy_to_coco(k['xyxy'])
        # new_annotations.append({
        #     "id": ann_id_counter,
        #     "image_id": img_id,
        #     "category_id": k['category_id'],
        #     "bbox": coco_box,
        #     "score": final_score, # 存入调整后的分数
        #     "area": round(coco_box[2] * coco_box[3], 2),
        #     "iscrowd": 0
        # })
        ann_id_counter += 1
#版本1
    # for k in final_keep:
    #     coco_box = xyxy_to_coco(k['xyxy'])
    #     new_annotations.append({
    #         "id": ann_id_counter,
    #         "image_id": img_id,
    #         "category_id": k['category_id'],
    #         "bbox": coco_box,
    #         "score": k['score'],
    #         "area": round(coco_box[2] * coco_box[3], 2),
    #         "iscrowd": 0
    #     })
    #     ann_id_counter += 1

# --- 6. 最终导出 ---
base_coco_data['annotations'] = new_annotations
with open(OUTPUT_PATH, 'w') as f:
    json.dump(base_coco_data, f, indent=4)

print(f"Done! Saved to {OUTPUT_PATH}. Total objects: {len(new_annotations)}")
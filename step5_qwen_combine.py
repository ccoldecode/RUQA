import json
import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
# ================= 1. 基础工具函数 =================
def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def xyxy_to_coco(xyxy):
    # 使用 round 避免浮点数精度导致的“微小新框”
    return [round(xyxy[0], 2), round(xyxy[1], 2), round(xyxy[2] - xyxy[0], 2), round(xyxy[3] - xyxy[1], 2)]

def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ================= 2. VLM 裁决核心 =================
def run_vlm_judge(model, processor, img_path, candidates):
    if not candidates: return []
    
    # 构建 Prompt 信息
    cand_info = "\n".join([f"- ID {c['temp_id']}: {c['label']}" for c in candidates])
#     prompt = f"""You are an Expert Marine Biologist. Evaluate if the candidate boxes contain real marine organisms.
# - KEEP: The box contains a real biological object (even if blurry).
# - REJECT: The box is noise, water artifacts, or empty background.
# [Candidates]
# {cand_info}
# Output JSON format: [ {{"id": "A", "verdict": "KEEP"}}, ... ]"""
    # 建议在 JSON 格式中加入 "reasoning" 字段，这能强制模型执行知识链推理过程
    prompt = f"""You are an Expert Marine Biologist specializing in underwater surveys. 
Your task is to verify if the objects within the candidate boxes are real marine organisms or just image artifacts (noise, shadows, or light refractions caused by turbidity).

[Biological Knowledge Base]
- Sea Urchin (Echinus): Look for spherical shapes with radiating spines or dark textured clusters.
- Sea Cucumber (Holothurian): Look for elongated, tubular bodies with bumpy, leathery skin. Often mimics rocks.
- Scallop: Look for fan-shaped shells with distinct radial ribs or slightly open shell gaps.
- Starfish: Look for pentagonal or radial symmetry with distinct arms and textured surfaces.

[Reasoning Chain]
For each box, follow these steps:
1. Feature Extraction: Identify morphological traits (e.g., symmetry, texture, appendages).
2. Artifact Filtering: Check if the object is too chaotic/geometric to be biological (e.g., sensor noise, backscatter).
3. Final Decision: Synthesize features to decide if it's a KEEP or REJECT.

[Candidates]
{cand_info}

Output JSON format (ensure valid JSON): 
[
  {{
    "id": "A", 
    "analysis": "Briefly describe biological features found (e.g., radial spines detected).", 
    "verdict": "KEEP"
  }},
  ...
]"""

    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    try:
        # 提取并解析 JSON
        start, end = output_text.find('['), output_text.rfind(']') + 1
        verdicts = json.loads(output_text[start:end])
        return {v['id']: v['verdict'] for v in verdicts if 'id' in v}
    except:
        return {}

# ================= 3. 主处理脚本 =================
def main():
    # --- 配置区 ---
    # 路径配置
    MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b" 
    IMG_ROOT = "/root/autodl-tmp/bi/ROUD/color"
    # ORIGINAL_JSON = "/root/autodl-tmp/bi/ROUD5_step4_blur.json"
    # DTIUIE_JSON = "/root/autodl-tmp/bi/ROUD5_step4_blurwE.json" # 请替换为真实路径
    # DGUNET_JSON = "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json" # 请替换为真实路径
    JSON_PATHS = {
        "original": "/root/autodl-tmp/bi/ROUD5_step4_color.json",
        "dtiuie": "/root/autodl-tmp/bi/ROUD5_step4_colorwE.json",
        "dgunet": "/root/autodl-tmp/bi/ROUD5_step4_colorwEdgunet.json"
    }
    OUTPUT_JSON = "/root/autodl-tmp/bi/ROUD5_step5_color2.json"

    # --- 模型加载 ---
    print("Loading Qwen2.5-VL...")
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

    # --- 数据加载与整合 ---
    raw_data = {k: json.load(open(v)) for k, v in JSON_PATHS.items()}
    id_to_img = {img['id']: img for img in raw_data['original']['images']}
    ID_TO_NAME = {cat['id']: cat['name'] for cat in raw_data['original']['categories']}
    
    all_anns_by_img = {}
    for src_name, data in raw_data.items():
        for ann in data['annotations']:
            img_id = ann['image_id']
            ann['source_file'] = src_name # 标记来源
            all_anns_by_img.setdefault(img_id, []).append(ann)

    new_annotations = []
    ann_id_counter = 1

    # --- 循环处理 ---
    for img_id, anns in tqdm(all_anns_by_img.items()):
        img_info = id_to_img.get(img_id)
        if not img_info: continue
        img_path = os.path.join(IMG_ROOT, os.path.basename(img_info['file_name']))
        if not os.path.exists(img_path): continue

        # A. 聚类：寻找共识
        clusters = []
        for ann in anns:
            box_xyxy = coco_to_xyxy(ann['bbox'])
            matched = False
            for cluster in clusters:
                if get_iou(box_xyxy, cluster['rep_box']) > 0.7:
                    cluster['anns'].append(ann)
                    cluster['sources'].add(ann['source_file'])
                    matched = True
                    break
            if not matched:
                clusters.append({'rep_box': box_xyxy, 'anns': [ann], 'sources': {ann['source_file']}})

        # B. 准备待审列表与数据防火墙（Snapshot）
        to_vlm_queue = []
        snapshot_pool = {} # 关键：存储原始数据

        # B. 准备待审列表与数据防火墙
        to_vlm_queue = []
        snapshot_pool = {}

        for i, cluster in enumerate(clusters):
            # --- 改进部分：不再单纯取 max，而是做加权融合 ---
            
            # 1. 坐标融合 (Spatial Fusion)
            # 把这个簇里所有框的坐标取平均值，得到一个更稳的“共识坐标”
            all_bboxes = [ann['bbox'] for ann in cluster['anns']]
            avg_bbox = [
                round(sum(b[0] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[1] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[2] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[3] for b in all_bboxes) / len(all_bboxes), 2)
            ]

            # 2. 标签投票 (Category Voting)
            # 如果不同模型对同一个位置认定的类别不同，取出现次数最多的那个
            all_cat_ids = [ann['category_id'] for ann in cluster['anns']]
            best_cat_id = max(set(all_cat_ids), key=all_cat_ids.count)
            
            # 3. 置信度综合
            # 既然大家都看到了，置信度应该取均值或最高值（因为你的值都很高，取均值更稳）
            avg_score = sum(ann.get('score', 0.9) for ann in cluster['anns']) / len(cluster['anns'])

            temp_id = chr(65 + i) if i < 26 else f"Z{i}"
            
            # 锁死这个融合后的数据
            box_data = {
                'temp_id': temp_id,
                'bbox': avg_bbox, 
                'category_id': best_cat_id,
                'score': round(avg_score, 4),
                'label': ID_TO_NAME.get(best_cat_id, "unknown")
            }
            # --- 改进结束 ---

            if len(cluster['sources']) >= 2:
                # 【共识框】：直接通过
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": box_data['category_id'], # 锁定
                    "bbox": box_data['bbox'],               # 锁定
                    "score": box_data['score'],
                    "area": round(box_data['bbox'][2] * box_data['bbox'][3], 2),
                    "iscrowd": 0
                })
                ann_id_counter += 1
            else:
                # 【存疑框】：存入 VLM 队列
                to_vlm_queue.append(box_data)
                snapshot_pool[temp_id] = box_data

        # C. 执行 VLM 裁决
        if to_vlm_queue:
            # 你可以尝试修改这里传入增强图的路径，或者直接传原图 img_path
            verdicts = run_vlm_judge(model, processor, img_path, to_vlm_queue)
            
            for t_id, v_res in verdicts.items():
                if v_res == "KEEP" and t_id in snapshot_pool:
                    orig = snapshot_pool[t_id]
                    new_annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": orig['category_id'], # 严格锁定原始类别
                        "bbox": orig['bbox'],               # 严格锁定原始坐标
                        "score": round(orig['score'] * 0.85, 4), # 软加权
                        "area": round(orig['bbox'][2] * orig['bbox'][3], 2),
                        "iscrowd": 0
                    })
                    ann_id_counter += 1

    # --- 最终导出 ---
    final_output = raw_data['original'].copy()
    final_output['annotations'] = new_annotations
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Processing Complete. New annotations: {len(new_annotations)}")

if __name__ == "__main__":
    main()
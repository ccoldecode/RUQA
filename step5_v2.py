import json
import os
import torch
import gc
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 1. 基础与图像工具 =================
def coco_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def xyxy_to_coco(xyxy):
    return [round(xyxy[0], 2), round(xyxy[1], 2), round(xyxy[2] - xyxy[0], 2), round(xyxy[3] - xyxy[1], 2)]

def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# def crop_with_margin(img, coco_bbox, margin=0.3):
#     """
#     最严格的裁剪函数：处理 COCO 格式 [x, y, w, h] 并增加边距
#     """
#     W, H = img.size
    
#     # 1. 确保输入是 COCO 格式 [x, y, w, h]
#     # 有些模型可能输出了负的 w 或 h，先取绝对值
#     x_min, y_min, w, h = coco_bbox
#     w, h = abs(w), abs(h)
    
#     # 2. 计算 xyxy 基础坐标
#     x1, y1 = x_min, y_min
#     x2, y2 = x_min + w, y_min + h
    
#     # 3. 施加 Margin (向外扩展)
#     x1 = x1 - w * margin
#     y1 = y1 - h * margin
#     x2 = x2 + w * margin
#     y2 = y2 + h * margin
    
#     # 4. 【核心纠错】暴力排序：确保 x1 < x2 且 y1 < y2
#     # 这一步能彻底解决 "lower < upper" 的报错
#     left, upper = min(x1, x2), min(y1, y2)
#     right, lower = max(x1, x2), max(y1, y2)
    
#     # 5. 限制在图片边界内，并转为整数
#     left = max(0, int(left))
#     upper = max(0, int(upper))
#     right = min(W, int(right))
#     lower = min(H, int(lower))
    
#     # 6. 【兜底】如果裁剪区域变成了“一条线”或“一个点”
#     if right <= left: right = min(W, left + 1)
#     if lower <= upper: lower = min(H, upper + 1)
        
#     return img.crop((left, upper, right, lower))
#     import math

def crop_with_margin(img, coco_bbox, margin=0.3):
    W, H = img.size
    
    # 1. 提取并清理数据，防止 NaN 或 None
    try:
        x_min, y_min, w, h = [float(x) if x is not None else 0.0 for x in coco_bbox]
        # 如果是 NaN，强制转为 0
        x_min = 0.0 if math.isnan(x_min) else x_min
        y_min = 0.0 if math.isnan(y_min) else y_min
        w = 10.0 if (math.isnan(w) or w <= 0) else w
        h = 10.0 if (math.isnan(h) or h <= 0) else h
    except:
        x_min, y_min, w, h = 0, 0, 10, 10

    # 2. 计算 xyxy
    x1, y1 = x_min, y_min
    x2, y2 = x_min + w, y_min + h
    
    # 3. 施加 Margin
    x1, y1 = x1 - w * margin, y1 - h * margin
    x2, y2 = x2 + w * margin, y2 + h * margin
    
    # 4. 【核心纠错】暴力排序并强制转整型，确保数值合法
    left, upper = int(min(x1, x2)), int(min(y1, y2))
    right, lower = int(max(x1, x2)), int(max(y1, y2))
    
    # 5. 限制边界
    left = max(0, left)
    upper = max(0, upper)
    right = min(W, right)
    lower = min(H, lower)
    
    # 6. 【终极保险】如果 right 还是小于等于 left，强制给 1 像素
    if right <= left:
        right = min(W, left + 1)
    if lower <= upper:
        lower = min(H, upper + 1)
        
    return img.crop((left, upper, right, lower))
# ================= 2. VLM 局部特写裁决核心 =================
# def run_vlm_judge_roi(model, processor, img_path, candidates):
#     if not candidates: 
#         return {}

#     # 1. 打开原图
#     try:
#         source_img = Image.open(img_path).convert("RGB")
#     except Exception as e:
#         print(f"Error opening image {img_path}: {e}")
#         return {}

#     # 2. 构建知识链 Prompt 前言
#     prompt_intro = """You are an Expert Marine Biologist. Evaluate if the following cropped image patches contain real marine organisms.
# [Biological Knowledge Base]
# - Sea Urchin: Spherical, radiating spines or dark textured clusters.
# - Sea Cucumber: Elongated, tubular, leathery/bumpy skin. Mimics rocks.
# - Scallop: Fan-shaped shell, radial ribs, often half-buried.
# - Starfish: Pentagonal/radial symmetry, arms.

# [Reasoning Chain]
# For EACH image patch, identify morphological traits. Filter out geometric sensor noise or water backscatter. Give a final verdict.
# - KEEP: Contains a real biological object.
# - REJECT: Noise, artifacts, or empty background."""

#     # 3. 构建多模态交错输入 (Interleaved Image-Text)
#     content_list = [{"type": "text", "text": prompt_intro}]
    
#     for c in candidates:
#         # 裁剪出带有 30% 边缘的特写图
#         patch_img = crop_with_margin(source_img, coco_to_xyxy(c['bbox']), margin=0.3)
        
#         # 将特写图按顺序加入对话
#         content_list.append({"type": "text", "text": f"\n--- Candidate ID: {c['temp_id']} (Labeled as {c['label']}) ---"})
#         content_list.append({"type": "image", "image": patch_img})
        
#     content_list.append({
#         "type": "text", 
#         "text": """\nOutput your final answer in strict JSON format:
# [
#   {"id": "A", "analysis": "brief reasoning...", "verdict": "KEEP"},
#   ...
# ]"""
#     })

#     messages = [{"role": "user", "content": content_list}]
    
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, _ = process_vision_info(messages)
    
#     # 动态分辨率控制：因为已经是特写图了，不需要太高分辨率，限制显存
#     inputs = processor(
#         text=[text], 
#         images=image_inputs, 
#         padding=True, 
#         return_tensors="pt",
#         max_pixels=512 * 28 * 28 # 每个特写切片最高只占用很少的 Token
#     ).to("cuda")

#     # 推理
#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
#     output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     # 释放显存
#     del inputs, generated_ids, image_inputs
#     source_img.close()
    
#     # 解析 JSON
#     try:
#         start = output_text.find('[')
#         end = output_text.rfind(']') + 1
#         verdicts = json.loads(output_text[start:end])
#         return {v['id']: v['verdict'] for v in verdicts if 'id' in v and 'verdict' in v}
#     except Exception as e:
#         print(f"JSON Parse Error for {img_path}. Model output: {output_text[:150]}")
#         return {}
# def run_vlm_judge_roi(model, processor, img_path, candidates):
#     if not candidates: 
#         return {}

#     try:
#         source_img = Image.open(img_path).convert("RGB")
#     except Exception as e:
#         print(f"Error opening image {img_path}: {e}")
#         return {}

#     # 1. 重新定义角色：将专家身份放入 system
#     system_prompt = "You are an Expert Marine Biologist. Respond ONLY in valid JSON format."
    
#     # 2. 构建知识链 Prompt
#     prompt_intro = """Verify if the following image patches contain real marine organisms.
# - KEEP: Contains a real biological object.
# - REJECT: Noise, artifacts, or background.

# Reasoning: Identify morphological traits (spines, symmetry, texture) then decide.
# [Candidates]"""

#     content_list = [{"type": "text", "text": prompt_intro}]
#     for c in candidates:
#         patch_img = crop_with_margin(source_img, coco_to_xyxy(c['bbox']), margin=0.3)
#         content_list.append({"type": "text", "text": f"ID {c['temp_id']} (Labeled {c['label']}):"})
#         content_list.append({"type": "image", "image": patch_img})
    
#     content_list.append({"type": "text", "text": "Final Answer in JSON [{\"id\": \"...\", \"verdict\": \"KEEP/REJECT\"}]:"})

#     # 3. 严格遵循 Qwen2.5-VL 的消息结构
#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
#         {"role": "user", "content": content_list}
#     ]
    
#     # 准备输入
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, _ = process_vision_info(messages)
    
#     inputs = processor(
#         text=[text], 
#         images=image_inputs, 
#         padding=True, 
#         return_tensors="pt",
#         max_pixels=512 * 28 * 28 
#     ).to("cuda")

#     # 4. 推理控制
#     with torch.no_grad():
#         # 增加 repetition_penalty 防止复读 Prompt
#         generated_ids = model.generate(**inputs, max_new_tokens=1024, repetition_penalty=1.1)
    
#     # 仅获取模型生成的回答部分（剔除 Prompt）
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
#     # 清理显存
#     del inputs, generated_ids, image_inputs
#     source_img.close()
    
#     # 5. 鲁棒的 JSON 解析
#     try:
#         # 去除 Markdown 标记
#         clean_text = output_text.replace("```json", "").replace("```", "").strip()
#         start = clean_text.find('[')
#         end = clean_text.rfind(']') + 1
#         if start == -1 or end == 0:
#             raise ValueError("No JSON array found")
        
#         verdicts = json.loads(clean_text[start:end])
#         return {v['id']: v['verdict'] for v in verdicts if 'id' in v and 'verdict' in v}
#     except Exception as e:
#         # 如果解析失败，打印前200个字符看看模型到底输出了什么
#         print(f"\n[Error] Image: {os.path.basename(img_path)} | Error: {e}")
#         print(f"[Raw Output Preview]: {output_text[:200]}")
#         return {}

def run_vlm_judge_roi(model, processor, img_path, candidates, batch_size=3):
    """
    预裁剪 + 小批次并行推理方案
    """
    if not candidates: 
        return {}

    try:
        source_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error: {e}")
        return {}

    # --- 1. 预裁剪操作：一次性把所有 Patch 准备好存入内存 ---
    # 这样后续推理时，CPU 就不再是瓶颈
    patches = []
    for c in candidates:
        patch = crop_with_margin(source_img, c['bbox'], margin=0.3)
        patches.append({
            'id': c['temp_id'],
            'label': c['label'],
            'image': patch
        })
    source_img.close() # 裁剪完即可关闭原图，节省内存

    verdict_map = {}
    system_prompt = "You are an Expert Marine Biologist. Respond ONLY in valid JSON format."

    # --- 2. 分批次推理 (Mini-Batch) ---
    for i in range(0, len(patches), batch_size):
        batch = patches[i : i + batch_size]
        
        # 构建当前批次的 Multi-modal Content
        content_list = []
        for p in batch:
            content_list.append({"type": "text", "text": f"Analyze ID {p['id']} ({p['label']}):"})
            content_list.append({"type": "image", "image": p['image']})
        
        content_list.append({"type": "text", "text": "Answer in JSON: [{\"id\": \"...\", \"verdict\": \"KEEP/REJECT\"}]"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content_list}
        ]

        # 3. 准备输入：大幅调低像素限制以换取极致速度
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt",
            # 特写图很小，224x224 级别的像素（16个Token）足够判断生物特征
            min_pixels=128 * 28 * 28, 
            max_pixels=256 * 28 * 28 
        ).to("cuda")

        # 4. 快速推理
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128, # 限制生成长度
                use_cache=True,     # 开启 KV Cache
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # 5. 解析结果
        trimmed_ids = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        try:
            clean_text = output_text.replace("```json", "").replace("```", "").strip()
            start, end = clean_text.find('['), clean_text.rfind(']') + 1
            res_list = json.loads(clean_text[start:end])
            for res in res_list:
                if 'id' in res and 'verdict' in res:
                    verdict_map[res['id']] = res['verdict']
        except:
            pass
        
        # 6. 每批次结束后强力释放显存
        del inputs, generated_ids, image_inputs
        torch.cuda.empty_cache()

    return verdict_map
# ================= 3. 主处理流水线 =================
def main():
    MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b" 
    IMG_ROOT = "/root/autodl-tmp/bi/ROUD/blur"
    OUTPUT_JSON = "/root/autodl-tmp/bi/ROUD5_step5_blur3.json"
    
    JSON_PATHS = {
        "original": "/root/autodl-tmp/bi/ROUD5_step4_blur.json",
        "dtiuie":   "/root/autodl-tmp/bi/ROUD5_step4_blurwE.json",
        "dgunet":   "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json"
    }
    print("Loading Qwen2.5-VL with use_fast=True...")
    # 建议配合 4-bit 量化使用，显存更稳
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        # min_pixels=MIN_PIXELS,
        # max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    raw_data = {k: json.load(open(v)) for k, v in JSON_PATHS.items()}
    id_to_img = {img['id']: img for img in raw_data['original']['images']}
    ID_TO_NAME = {cat['id']: cat['name'] for cat in raw_data['original']['categories']}
    
    all_anns_by_img = {}
    for src_name, data in raw_data.items():
        for ann in data['annotations']:
            img_id = ann['image_id']
            ann['source_file'] = src_name
            all_anns_by_img.setdefault(img_id, []).append(ann)

    new_annotations = []
    ann_id_counter = 1

    print("Starting Processing...")
    for img_id, anns in tqdm(all_anns_by_img.items()):
        img_info = id_to_img.get(img_id)
        if not img_info: continue
        img_path = os.path.join(IMG_ROOT, os.path.basename(img_info['file_name']))
        if not os.path.exists(img_path): continue

        # A. 共识聚类 (WBF 简化版)
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

        # B. 数据分流与特征融合
        to_vlm_queue = []
        snapshot_pool = {} 

        for i, cluster in enumerate(clusters):
            # 坐标平均融合
            all_bboxes = [ann['bbox'] for ann in cluster['anns']]
            avg_bbox = [
                round(sum(b[0] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[1] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[2] for b in all_bboxes) / len(all_bboxes), 2),
                round(sum(b[3] for b in all_bboxes) / len(all_bboxes), 2)
            ]
            
            # 类别投票
            all_cat_ids = [ann['category_id'] for ann in cluster['anns']]
            best_cat_id = max(set(all_cat_ids), key=all_cat_ids.count)
            avg_score = sum(ann.get('score', 0.9) for ann in cluster['anns']) / len(cluster['anns'])

            temp_id = chr(65 + i) if i < 26 else f"Z{i}"
            
            box_data = {
                'temp_id': temp_id,
                'bbox': avg_bbox, 
                'category_id': best_cat_id,
                'score': round(avg_score, 4),
                'label': ID_TO_NAME.get(best_cat_id, "unknown")
            }

            if len(cluster['sources']) >= 2:
                # 达成共识：直接保存
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": box_data['category_id'],
                    "bbox": box_data['bbox'],
                    "score": box_data['score'],
                    "area": round(box_data['bbox'][2] * box_data['bbox'][3], 2),
                    "iscrowd": 0
                })
                ann_id_counter += 1
            else:
                # 存在争议：送入特写裁决队列
                to_vlm_queue.append(box_data)
                snapshot_pool[temp_id] = box_data

        # C. 运行 VLM 局部特写验证
        if to_vlm_queue:
            verdicts = run_vlm_judge_roi(model, processor, img_path, to_vlm_queue)
            
            for t_id, v_res in verdicts.items():
                if v_res == "KEEP" and t_id in snapshot_pool:
                    orig = snapshot_pool[t_id]
                    new_annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": orig['category_id'],
                        "bbox": orig['bbox'],
                        "score": round(orig['score'] * 0.85, 4), # VLM保底的分数略降
                        "area": round(orig['bbox'][2] * orig['bbox'][3], 2),
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
        
        # 定期清理显存，防止长循环累积爆炸
        torch.cuda.empty_cache()
        gc.collect()

    # --- 最终导出 ---
    final_output = raw_data['original'].copy()
    final_output['annotations'] = new_annotations
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Processing Complete. New annotations: {len(new_annotations)}")

if __name__ == "__main__":
    main()
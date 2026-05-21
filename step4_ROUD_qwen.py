import torch
import os
import json
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== 配置 ====================
MODEL_DIR = '/root/autodl-tmp/bi/qwen2.5-vl-7b'   # ← 请修改成你实际的 Qwen 模型路径
BASE_DIR = "/root/autodl-tmp/bi/ROUD"
DIAGNOSIS_JSON = "ROUD_step2_fixed.json"
GT_DIR = "/root/autodl-tmp/bi/results"

SUBSETS = {
    "blur": "instances_blur.json",
}

TARGET_CATEGORIES = [
    {"id": 1, "name": "holothurian"}, {"id": 2, "name": "echinus"},
    {"id": 3, "name": "scallop"}, {"id": 4, "name": "starfish"},
    {"id": 5, "name": "fish"}, {"id": 6, "name": "corals"},
    {"id": 7, "name": "diver"}, {"id": 8, "name": "cuttlefish"},
    {"id": 9, "name": "turtle"}, {"id": 10, "name": "jellyfish"}
]

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# ==================== 加载诊断信息 ====================
def load_diagnosis_map(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    diag_map = {}
    for item in data:
        fname = os.path.basename(item.get('image_path', ''))
        hybrid = item.get('hybrid_evaluation', {})
        diag_map[fname] = {
            "L": hybrid.get('turbidity_blur', {}).get('score', 5.0),
            "C": 5.0,
            "B": 5.0
        }
    return diag_map

# ==================== 加强版 Prompt（重点防重复） ====================
def build_prompt(diag):
    prompt = f"""Visual Context: Turbidity {diag['L']:.1f}/10.

You are an expert underwater marine life detector.

Detect ALL visible instances of these categories ONLY: 
holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.

Important Rules:
- Each object should be detected **only once**. Do NOT generate multiple overlapping or stacked boxes for the same object.
- Do not repeat the same fish many times.
- If several fish are close together, use one reasonable bbox instead of many tiny ones.
- Output **ONLY** a valid JSON array. No explanation, no markdown, no extra text.

Format:
[
  {{ "bbox_2d": [ymin, xmin, ymax, xmax], "label": "fish" }}
]

If no clear objects, output: []

Begin now:"""
    return prompt

# ==================== 主函数 ====================
def run_prediction_with_physics():
    print("🚀 正在加载 Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True
    )

    diag_map = load_diagnosis_map(DIAGNOSIS_JSON)
    cat_to_id = {cat['name'].lower(): cat['id'] for cat in TARGET_CATEGORIES}

    for folder, json_name in SUBSETS.items():
        img_dir = os.path.join(BASE_DIR, folder)
        save_path = f"step4_qwen_{json_name}"

        # 断点续跑
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                output_coco = json.load(f)
            print(f"✅ 继续运行，已处理 {len(output_coco.get('images', []))} 张")
        else:
            output_coco = {"images": [], "annotations": [], "categories": TARGET_CATEGORIES}

        ann_id_counter = len(output_coco.get("annotations", [])) + 1

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for img_name in tqdm(img_files, desc=f"Processing {folder}"):
            img_id = len(output_coco["images"]) + 1

            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                o_w, o_h = image.size
                r_h, r_w = smart_resize(o_h, o_w, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
                s_w = o_w / r_w
                s_h = o_h / r_h

                diag = diag_map.get(img_name, {"L": 5.0})
                prompt = build_prompt(diag)

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": prompt}
                ]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=1024, 
                        do_sample=False, 
                        repetition_penalty=1.25   # 稍微提高，减少重复
                    )
                    output_text = processor.batch_decode(
                        [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
                        skip_special_tokens=True
                    )[0]

                # 保存图像信息
                output_coco["images"].append({"id": img_id, "file_name": img_name, "width": o_w, "height": o_h})

                # ==================== 强力解析 + 强去重 ====================
                seen_boxes = set()
                try:
                    clean_text = re.sub(r'```json\s*|\s*```|```', '', output_text, flags=re.DOTALL).strip()
                    
                    json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', clean_text, re.DOTALL)
                    clean_json = json_match.group(0) if json_match else clean_text

                    # 修复常见 JSON 损坏
                    clean_json = re.sub(r',\s*(\]|\})', r'\1', clean_json)
                    clean_json = re.sub(r'"\s*,\s*}', '"}', clean_json)

                    detections = json.loads(clean_json)

                    for det in detections:
                        label = str(det.get("label", "")).lower().strip()
                        bbox = det.get("bbox_2d", det.get("bbox", []))
                        if len(bbox) != 4:
                            continue

                        cid = cat_to_id.get(label, -1)
                        if cid == -1:
                            continue

                        x1, y1, x2, y2 = [float(v) for v in bbox]
                        rx = round(max(0, min(x1, x2) * s_w), 2)
                        ry = round(max(0, min(y1, y2) * s_h), 2)
                        rw = round(abs(x2 - x1) * s_w, 2)
                        rh = round(abs(y2 - y1) * s_h, 2)

                        if rw <= 5 or rh <= 5:
                            continue

                        # 强去重：位置接近的视为同一个（解决大量重复框问题）
                        box_key = (cid, round(rx / 12), round(ry / 12))
                        if box_key not in seen_boxes:
                            seen_boxes.add(box_key)
                            output_coco.setdefault("annotations", []).append({
                                "id": ann_id_counter,
                                "image_id": img_id,
                                "category_id": cid,
                                "bbox": [rx, ry, rw, rh],
                                "area": round(rw * rh, 2),
                                "iscrowd": 0,
                                "score": 0.95
                            })
                            ann_id_counter += 1

                except Exception as e:
                    print(f"\n⚠️ 解析失败 {img_name}: {e}")
                    try:
                        with open(f"failed_{img_name}.txt", "w", encoding="utf-8") as f:
                            f.write(output_text)
                    except:
                        pass

                # 立即保存（断点续跑安全）
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(output_coco, f, indent=2, ensure_ascii=False)

                # 清理显存
                del image, inputs, generated_ids
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\n❌ 处理异常 {img_name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        print(f"\n✅ {folder} 处理完成！结果保存至: {save_path}")


if __name__ == "__main__":
    run_prediction_with_physics()
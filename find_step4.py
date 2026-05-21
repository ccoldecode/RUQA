import torch
import os
import json
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

# ================= 1. 配置路径 =================
MODEL_PATH = "/root/autodl-tmp/bi/qwen2.5-vl-7b"

# 需要从中读取提示词上下文的 Step2 JSON
STEP2_JSON_PATH = "/root/autodl-tmp/bi/all_step2_fixed.json"

# 之前跑完、但有少量失败的 Step4 JSON
STEP4_ORIGINAL_JSON = "/root/autodl-tmp/bi/all_step4wE.json"

# 补考并合并后输出的新 JSON
STEP4_MERGED_JSON = "/root/autodl-tmp/bi/all_step4wE_fixed.json"

RAW_IMAGE_DIR = "/root/autodl-tmp/bi/data/dataset"
ENHANCED_IMAGE_DIR = "/root/autodl-tmp/bi/DTIUIE_output2"

USE_ENHANCED = False   # ← 改成 True 就是用增强图

CATEGORY_MAP = {
    "sea cucumber": 1, "sea urchin": 2, "scallop": 3, "starfish": 4, "fish": 5,
    "corals": 6, "diver": 7, "squid": 8, "turtle": 9, "jellyfish": 10
}

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

# ================= 2. 强力 JSON 解析器 =================
def extract_and_parse_json(text):
    """提取模型输出中的 JSON，并尝试修复被截断的结尾"""
    # 去除 Markdown 代码块标记
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    
    # 尝试找到最外层的列表 []
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    json_str = match.group(1) if match else text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 如果解析失败，尝试暴力补齐右括号 ]
        try:
            if json_str.strip().startswith('[') and not json_str.strip().endswith(']'):
                fixed_str = re.sub(r',[^{]*$', '', json_str) + ']'
                return json.loads(fixed_str)
        except:
            pass
        return None

# ================= 3. 构建 Prompt 函数 =================
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

# ================= 4. 核心补丁与合并逻辑 =================
def main():
    print(f"📦 正在加载诊断上下文: {STEP2_JSON_PATH}")
    with open(STEP2_JSON_PATH, 'r', encoding='utf-8') as f:
        step2_data = json.load(f)
    # 将 step2 数据转换为以文件名为 key 的字典，方便查询
    step2_dict = {os.path.basename(item.get('image_path', '')): item for item in step2_data}

    print(f"📦 正在加载已标注数据: {STEP4_ORIGINAL_JSON}")
    with open(STEP4_ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        step4_data = json.load(f)

    images_info = step4_data.get('images', [])
    annotations = step4_data.get('annotations', [])

    # 获取已成功标注的 image_id
    annotated_image_ids = {ann['image_id'] for ann in annotations if 'image_id' in ann}

    # 找出解析失败（即没有对应标注）的图片
    failed_images = [img for img in images_info if img['id'] not in annotated_image_ids]
    
    print(f"\n📊 统计：总计 {len(images_info)} 张，已成功 {len(annotated_image_ids)} 张，需要补跑 {len(failed_images)} 张。")

    if len(failed_images) == 0:
        print("✅ 所有图片均已成功标注，不需要补跑！")
        return

    # 获取当前最大的 annotation ID，防止新加的 ID 冲突
    ann_id = max([a['id'] for a in annotations] + [0]) + 1

    # ================= 加载模型 =================
    print(f"\n🚀 开始加载 Qwen 模型 (补考模式)...")
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

    new_success_count = 0

    # 遍历需要补考的图片
    for img_dict in tqdm(failed_images, desc="Patching Missing Data"):
        img_id = img_dict['id']
        img_name = img_dict['file_name']
        img_path = os.path.join(ENHANCED_IMAGE_DIR if USE_ENHANCED else RAW_IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"\n⚠️ 找不到图片: {img_path}，跳过")
            continue

        # 获取对应的 step2 上下文
        step2_entry = step2_dict.get(img_name, {})

        # 计算缩放系数 (沿用你之前的逻辑)
        img_obj = Image.open(img_path)
        o_w, o_h = img_obj.size
        r_h, r_w = smart_resize(o_h, o_w, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        s_w = o_w / r_w
        s_h = o_h / r_h

        # 推理
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": build_prior_prompt(step2_entry)}
        ]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,  # 🔥 改为 2048，防止 4K 大图目标太多被截断
                do_sample=False,
                repetition_penalty=1.15
            )
            output_text = processor.batch_decode(
                [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]

        # 解析输出
        detections = extract_and_parse_json(output_text)

        if detections is not None:
            seen_boxes = set()
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
                            # 加入原有的 annotations 列表
                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cid,
                                "bbox": [rx, ry, rw, rh],
                                "score": 0.95,
                                "area": round(rw * rh, 2),
                                "iscrowd": 0
                            })
                            ann_id += 1
            new_success_count += 1
        else:
            print(f"\n❌ [{img_name}] 终极解析失败。模型原始输出如下:\n{output_text[:200]}...")

    # ================= 5. 保存最终合并结果 =================
    step4_data['annotations'] = annotations

    with open(STEP4_MERGED_JSON, "w", encoding='utf-8') as f:
        # 为了速度和体积，这里不再用 indent=4，如果你需要易读格式，可以加回来
        json.dump(step4_data, f, ensure_ascii=False)

    print(f"\n🎉 合并完成！本次成功补考了 {new_success_count} 张图片。")
    print(f"📁 最终完整标注文件已保存至: {STEP4_MERGED_JSON}")

if __name__ == '__main__':
    main()
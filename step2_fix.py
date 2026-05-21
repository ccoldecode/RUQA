import torch
from PIL import Image
import time
import re
import json
import os
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

# ==================== 核心组件 1：极端容错 JSON 解析器 ====================
def robust_json_decode(text: str) -> dict:
    """暴力提取并尝试修复截断的 JSON"""
    # 1. 提取大括号内容
    match = re.search(r'(\{.*\})', text.replace('\n', ' '), re.DOTALL)
    if not match:
        return {}
    
    clean_text = match.group(1)
    
    # 2. 正常解析尝试
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass
    
    # 3. 截断修复尝试 (穷举补全右括号和引号)
    suffixes = ['}', '"}', '"]}', '}}', '"}}', '"]}}']
    for suffix in suffixes:
        try:
            return json.loads(clean_text + suffix)
        except:
            continue
            
    # 4. 如果还是失败，尝试用正则强行挖出关键字段 (终极保底)
    fallback_json = {"tags": {}, "severity_scores": {}}
    
    t_match = re.search(r'"turbidity"\s*:\s*"([L123]+)"', clean_text)
    if t_match: fallback_json["tags"]["turbidity"] = t_match.group(1)
    
    c_match = re.search(r'"color_cast"\s*:\s*"([C123]+)"', clean_text)
    if c_match: fallback_json["tags"]["color_cast"] = c_match.group(1)
    
    b_match = re.search(r'"brightness"\s*:\s*"([B123]+)"', clean_text)
    if b_match: fallback_json["tags"]["brightness"] = b_match.group(1)
    
    return fallback_json if fallback_json["tags"] else {}

# ==================== 核心组件 2：分数与标签一致性强制对齐 ====================
def enforce_score_alignment(vlm_json: dict) -> dict:
    """
    检查并强制修正分数与标签不匹配、或全为0的情况。
    用户定义：L1/C1/B1 (0-2), L2/C2/B2 (3-7), L3/C3/B3 (8-10)
    """
    if "tags" not in vlm_json:
        vlm_json["tags"] = {}
    if "severity_scores" not in vlm_json:
        vlm_json["severity_scores"] = {}

    tags = vlm_json.get("tags", {})
    scores = vlm_json.get("severity_scores", {})

    # 定义安全的默认锚点分值
    safe_anchors = {"1": 1.5, "2": 5.0, "3": 9.0}
    
    mappings = [
        ("turbidity", "turbidity_score", "L"),
        ("color_cast", "color_score", "C"),
        ("brightness", "brightness_score", "B")
    ]

    for tag_key, score_key, prefix in mappings:
        current_tag = tags.get(tag_key, f"{prefix}2") # 缺省给中等
        current_score = scores.get(score_key, 0.0)
        
        # 提取标签级别 (1, 2, 或 3)
        level = "2"
        match = re.search(r'\d', current_tag)
        if match:
            level = match.group()

        # 检查是否匹配逻辑边界
        is_mismatched = False
        if level == "1" and not (0 <= current_score <= 2.5): is_mismatched = True
        elif level == "2" and not (2.6 <= current_score <= 7.5): is_mismatched = True
        elif level == "3" and not (7.6 <= current_score <= 10.0): is_mismatched = True
        elif current_score == 0.0: is_mismatched = True # 绝对0分也视为偷懒

        # 如果发生不匹配，使用 Python 强制覆盖，不给模型狡辩的机会
        if is_mismatched:
            scores[score_key] = safe_anchors.get(level, 5.0)

    # 生成 combined_tag 以防丢失
    t = tags.get("turbidity", "L2")
    c = tags.get("color_cast", "C2")
    b = tags.get("brightness", "B2")
    tags["combined_tag"] = f"{t}-{c}-{b}"

    vlm_json["tags"] = tags
    vlm_json["severity_scores"] = scores
    return vlm_json

# ==================== 核心组件 3：诊断器 ====================
def is_bad_data(item: dict) -> bool:
    eval_res = item.get("hybrid_evaluation", {})
    if not eval_res or "tags" not in eval_res:
        return True
    
    tags = eval_res.get("tags", {})
    scores = eval_res.get("severity_scores", {})
    
    # 检查字段完整性
    if not tags or not scores: return True
    if "turbidity_score" not in scores or "color_score" not in scores: return True
    
    # 检查分数是否全为 0
    t_s = scores.get("turbidity_score", 0)
    c_s = scores.get("color_score", 0)
    b_s = scores.get("brightness_score", 0)
    if t_s == 0 and c_s == 0 and b_s == 0:
        return True

    # 检查严重不匹配（例如标签是 L3，分数却是 1.0）
    t_tag = tags.get("turbidity", "")
    if "3" in t_tag and t_s < 7.0: return True
    if "1" in t_tag and t_s > 3.0: return True
    
    c_tag = tags.get("color_cast", "")
    if "3" in c_tag and c_s < 7.0: return True
    if "1" in c_tag and c_s > 3.0: return True

    return False

# ==================== 加载模型 (仅在执行修复时加载) ====================
model = None
processor = None

def init_model():
    global model, processor
    print("🚀 正在加载 Qwen2.5-VL-7B-Instruct 修复专用实例...")
    model_path = "/root/autodl-tmp/bi/qwen2.5-vl-7b"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=512*28*28)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

# ==================== 修复专用推理逻辑 ====================
def repair_evaluate_batch(image_paths: list, items_data: list, batch_size: int):
    # 这里导入你的决策引擎逻辑（假设你在 step2_fast.py 中有这个函数）
    # 如果没有，请将 get_hybrid_decision 的代码复制到此脚本中
    from step2_fast import get_hybrid_decision, calculate_confidence_from_scores

    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_items = items_data[i:i + batch_size]
        
        batch_images = []
        batch_prompts = []
        
        for path, item in zip(batch_paths, batch_items):
            metrics = item.get("metrics", item)
            img = Image.open(path).convert("RGB")
            img.thumbnail((480, 480), Image.Resampling.LANCZOS)
            batch_images.append(img)
            
            # 【关键修改】：Prompt 中的示例不再使用 0.0，诱导模型输出真实数值
            # 在 repair_evaluate_batch 循环内部修改 prompt 定义
            prompt = f"""你是一位精通水下成像物理模型的资深视觉科学家。
请以图像的真实视觉特征为主，并结合传入的客观物理指标作为辅助印证，对水下图像进行物理分量拆解并给出 L-C-B 定量诊断。

### 1. 评分与标签定义 (0-10 刻度)
**注意：分数越高代表退化越严重。**
- **浑浊度 (L/Turbidity)**: 
  - [0-2] L1: 清晰，结构完整。
  - [3-7] L2: 轻微雾化，边缘开始模糊。
  - [8-10] L3: 浓重散射，细节大量丢失。
- **色偏 (C/Color)**: 
  - [0-2] C1: 色彩平衡自然。
  - [3-7] C2: 明显蓝色/青色偏。
  - [8-10] C3: 严重绿色/黄色/褐色偏。
- **亮度 (B/Brightness)**: 
  - [0-2] B1: 辐照度充足，画面明亮。
  - [3-7] B2: 能量受限，画面偏暗。
  - [8-10] B3: 极低照度，噪点主导。

### 2. 输入物理指标
- UCIQE: {metrics.get('uciqe', 'N/A')}
- UIQM: {metrics.get('uiqm', 'N/A')}
- ColorImbalance: {metrics.get('color_imbalance', 'N/A')}
- Brightness: {metrics.get('brightness', 'N/A')}
- LaplacianVar: {metrics.get('laplacian_var', 'N/A')}

### 3. 诊断约束
**请打破“中庸偏见”：** 如果画面清晰，请大胆给出 0-2 分；如果画面极其恶劣，请大胆给出 8-10 分。**禁止默认给出 5.0 分左右的中间值。**

### 4. 输出格式要求
必须且只能输出合法的 JSON 数据。禁止输出 Markdown 标记。
**注意：下方为输出格式示例，分值仅为演示，你必须根据实际特征输出真实浮点数（保留一位小数）。**

{{
  "reasoning": "必须首先在此字段引用具体指标与视觉特征，以此推导后续分值。",
  "tags": {{
    "turbidity": "L2",
    "color_cast": "C2",
    "color_type": "green",
    "brightness": "B1",
    "combined_tag": "L2-C2-B1"
  }},
  "severity_scores": {{
    "turbidity_score": 5.2, 
    "color_score": 3.8,
    "brightness_score": 1.5
  }},
  "physical_diagnostics": {{
    "dominant_degradation": "color",
    "needs_enhancement": true,
    "small_object_detectability": "medium"
  }},
  "vlm_confidence": 0.9
}}"""
            batch_prompts.append(prompt)

        messages_list = [[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}] for img, p in zip(batch_images, batch_prompts)]
        texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages_list)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            # 【关键修改】：max_new_tokens 提升到 768 防止截断
            output_dict = model.generate(
                **inputs, max_new_tokens=768, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True
            )

        generated_ids = [ids[input_len:] for ids in output_dict.sequences]
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
        vlm_confs = calculate_confidence_from_scores(output_dict.sequences, output_dict.scores, input_len, processor.tokenizer.pad_token_id)

        for j, (path, item, output_text) in enumerate(zip(batch_paths, batch_items, outputs)):
            metrics = item.get("metrics", item)
            
            # 使用超级解析器
            vlm_json = robust_json_decode(output_text)
            raw_json_failed = False if vlm_json else True
            
            # 无论如何，强制对齐分数和标签
            vlm_json = enforce_score_alignment(vlm_json)
            vlm_json["vlm_confidence"] = round(vlm_confs[j], 4)

            decision = get_hybrid_decision(vlm_json, metrics, raw_json_failed=raw_json_failed)
            
            if raw_json_failed:
                print(f"🚑 触发底线保底修复 → {os.path.basename(path)}")
            else:
                print(f"✅ 完美修复：{os.path.basename(path)} | LCB: {vlm_json['tags'].get('combined_tag')}")

            results.append({"image_path": path, "metrics": metrics, "hybrid_evaluation": decision, "processing_time": 0.0})

    return results

# ==================== 主控引擎 ====================
def repair_pipeline(input_json_path: str = "DUO_step2.json", output_json_path: str = "DUO_step2_fixed.json", batch_size: int = 4):
    if not os.path.exists(input_json_path): return
    with open(input_json_path, "r", encoding="utf-8") as f: all_data = json.load(f)

    valid_results, need_fix_items = [], []
    for item in all_data:
        if is_bad_data(item): need_fix_items.append({"image_path": item["image_path"], "metrics": item["metrics"]})
        else: valid_results.append(item)

    print(f"📊 扫描完成：正常 {len(valid_results)} 条，异常 {len(need_fix_items)} 条待修复。")
    if not need_fix_items: return

    init_model() # 仅在有任务时占用显存

    fixed_results = []
    fix_paths = [x["image_path"] for x in need_fix_items]
    fix_metadata = [x["metrics"] for x in need_fix_items]

    for i in tqdm(range(0, len(fix_paths), batch_size), desc="强力修复中"):
        batch_p = fix_paths[i:i+batch_size]
        batch_m = fix_metadata[i:i+batch_size]
        batch_results = repair_evaluate_batch(batch_p, batch_m, batch_size)
        fixed_results.extend(batch_results)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(valid_results + fixed_results, f, indent=2, ensure_ascii=False)
    print(f"🎊 修复完成！修正后的数据已保存至: {output_json_path}")

if __name__ == "__main__":
    repair_pipeline()
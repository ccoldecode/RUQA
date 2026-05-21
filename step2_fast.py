import torch
from PIL import Image
import json
from tqdm import tqdm
import time
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
# ==================== 环境配置 ====================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "8"

print("🚀 正在加载 Qwen2.5-VL-7B-Instruct 模型...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/bi/qwen2.5-vl-7b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/bi/qwen2.5-vl-7b",
    trust_remote_code=True,
    padding_side="left"
)

processor = AutoProcessor.from_pretrained(
    "/root/autodl-tmp/bi/qwen2.5-vl-7b",
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

# 【关键新增】强制让 processor 的 tokenizer 使用左侧填充，并确保 pad_token 存在
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
print("✅ 模型加载完成！已强制设定左侧 padding (Left-Padding)")

# ==================== 新增：基于 Logprobs 计算平均置信度 ====================
def calculate_confidence_from_scores(sequences, scores, input_len, pad_token_id):
    """
    基于模型输出的 logits 计算生成文本的平均概率
    """
    batch_size = sequences.shape[0]
    generated_ids = sequences[:, input_len:]
    gen_len = generated_ids.shape[1]
    
    confidences = []
    for b in range(batch_size):
        probs = []
        for t in range(gen_len):
            token_id = generated_ids[b, t]
            # 遇到 pad 符说明这句 JSON 已经生成完了
            if token_id == pad_token_id:
                break
            
            # 获取当前 Token 的概率
            logits = scores[t][b]
            token_prob = F.softmax(logits, dim=-1)[token_id].item()
            probs.append(token_prob)
        
        # 计算整句 JSON 的平均置信度
        if probs:
            confidences.append(sum(probs) / len(probs))
        else:
            confidences.append(0.0)
            
    return confidences
    
# ==================== Prompt（保留你原意，仅在最后加强格式） ====================
def get_underwater_prompt(metrics: dict) -> str:
    return f"""你是一位精通水下成像物理模型的资深视觉科学家。
请以图像的真实视觉特征为主，并结合传入的客观物理指标作为辅助印证，对水下图像进行物理分量拆解并给出 L-C-B 定量诊断。

### 1. 评分与标签定义 (0-10 刻度)
**注意：分数越高代表退化越严重。**
- **浑浊度 (L/Turbidity)**: 
  - [0-2] L1: 清晰，结构完整。
  - [3-7] L2: 轻微雾化，边缘开始模糊。
  - [8-10] L3: 浓重散射，幕帘效应严重，细节大量丢失。
- **色偏 (C/Color)**: 
  - [0-2] C1: 色彩平衡自然。
  - [3-7] C2: 明显蓝色/青色偏。
  - [8-10] C3: 严重绿色/黄色/褐色偏，固有色基本丧失。
- **亮度 (B/Brightness)**: 
  - [0-2] B1: 辐照度充足，画面明亮。
  - [3-7] B2: 能量受限，画面偏暗。
  - [8-10] B3: 极低照度，噪点主导，细节被噪声掩盖。

### 2. 输入物理指标
*注意：UCIQE/UIQM 越高代表质量越好，对应的退化分应越低。*
- UCIQE: {metrics.get('uciqe', 'N/A')}
- UIQM: {metrics.get('uiqm', 'N/A')}
- ColorImbalance: {metrics.get('color_imbalance', 'N/A')}
- Brightness: {metrics.get('brightness', 'N/A')}
- LaplacianVar: {metrics.get('laplacian_var', 'N/A')}

### 3. 诊断约束（核心提示）
**请打破“中庸偏见”：** 如果画面清晰，请大胆给出 0-2 分；如果画面极度模糊或色偏严重，请大胆给出 8-10 分。**严禁在未经过细节比对的情况下默认给出 5 分左右的中间值。**

### 4. 输出格式要求
必须且只能输出合法的 JSON 数据。禁止输出任何解释性文字或 Markdown 标记。
**注意：必须严格按照下方 JSON 键的顺序输出，先进行推理阐述，再进行打分。**

{{
  "reasoning": "必须首先在此字段引用具体指标与视觉特征（如：虽然UCIQE尚可，但视觉观察到严重绿色偏，推测为C3级别...），以此推导后续分值。",
  "tags": {{
    "turbidity": "L1/L2/L3",
    "color_cast": "C1/C2/C3",
    "color_type": "none/blue/green/cyan/yellow/brown",
    "brightness": "B1/B2/B3",
    "combined_tag": "L_-C_-B_"
  }},
  "severity_scores": {{
    "turbidity_score": 0.0, 
    "color_score": 0.0,
    "brightness_score": 0.0
  }},
  "physical_diagnostics": {{
    "dominant_degradation": "color | turbidity | brightness",
    "needs_enhancement": true,
    "small_object_detectability": "high/medium/low"
  }},
  "vlm_confidence": 0.9
}}"""

# ==================== 混合决策引擎（不变） ====================
def get_hybrid_decision(vlm_data: dict, metrics: dict, raw_json_failed: bool = False) -> dict:
    thresholds = {
        'color_imbalance': 0.12,
        'brightness': 85,
        'laplacian_var': 180,
        'uiqm': 1.8,
        'uciqe': 0.45
    }
    def rule_check(m):
        conditions = [
            m.get('color_imbalance', 0) > thresholds['color_imbalance'],
            m.get('brightness', 255) < thresholds['brightness'],
            m.get('laplacian_var', 999) < thresholds['laplacian_var']
        ]
        quality_low = (m.get('uiqm', 99) < thresholds['uiqm']) or (m.get('uciqe', 99) < thresholds['uciqe'])
        return sum(conditions) >= 2 or quality_low
    rule_says_yes = rule_check(metrics)

    if raw_json_failed or vlm_data.get('vlm_confidence', 0) < 0.7:
        needs_enhancement = rule_says_yes
        final_source = "RULE_FALLBACK_LOW_CONF"
        enhancement_priority = "combined"
        physical = vlm_data.get("physical_diagnostics", {})
    elif vlm_data.get('vlm_confidence', 0) >= 0.75:
        physical = vlm_data.get("physical_diagnostics", {})
        vlm_says_yes = bool(physical.get("needs_enhancement", False))
        needs_enhancement = vlm_says_yes
        final_source = "VLM"
        enhancement_priority = physical.get("dominant_degradation", "combined")
    else:
        physical = vlm_data.get("physical_diagnostics", {})
        vlm_says_yes = bool(physical.get("needs_enhancement", False))
        needs_enhancement = rule_says_yes or vlm_says_yes
        final_source = "RULE_OVERRIDE_CONSERVATIVE"
        if metrics.get('color_imbalance', 0) > thresholds['color_imbalance']:
            enhancement_priority = "color"
        elif metrics.get('laplacian_var', 999) < thresholds['laplacian_var']:
            enhancement_priority = "turbidity"
        else:
            enhancement_priority = "combined"

    if not needs_enhancement and rule_says_yes and final_source == "VLM":
        needs_enhancement = True
        final_source = "RULE_OVERRIDE_CONSERVATIVE"

    return {
        "needs_enhancement": needs_enhancement,
        "final_decision_source": final_source,
        "enhancement_priority": enhancement_priority,
        "vlm_confidence": vlm_data.get("vlm_confidence", 0.0),
        "tags": vlm_data.get("tags", {}),
        "severity_scores": vlm_data.get("severity_scores", {}),
        "physical_diagnostics": physical,
        "reasoning": vlm_data.get("reasoning", "Rule fallback triggered")
    }

# ==================== 批量评估（最强清理逻辑） ====================
# ==================== 批量评估（最强清理 + Logprobs 置信度） ====================
def evaluate_batch(image_paths: list, items_data: list, batch_size: int = 8):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_items = items_data[i:i + batch_size]

        batch_images = []
        batch_prompts = []
        for path, item in zip(batch_paths, batch_items):
            metrics = item.get("metrics", item) if isinstance(item, dict) else item
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((480, 480), Image.Resampling.LANCZOS)
                batch_images.append(img)
                batch_prompts.append(get_underwater_prompt(metrics))
            except Exception as e:
                print(f"❌ 加载图像失败 {path}: {e}")
                continue

        if not batch_images:
            continue

        messages_list = []
        for img, prompt in zip(batch_images, batch_prompts):
            messages_list.append([{"role": "user", "content": [
                {"type": "image", "image": img, "max_pixels": 512 * 28 * 28},
                {"type": "text", "text": prompt}
            ]}])

        texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages_list)

        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # 【关键改动 1】开启 output_scores 和 return_dict，并传入 pad_token_id
            output_dict = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                output_scores=True,           
                return_dict_in_generate=True  
            )

        # 【关键改动 2】提取生成的 ID 并计算真实置信度
        generated_ids_trimmed = [ids[input_len:] for ids in output_dict.sequences]
        outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        
        vlm_confidences = calculate_confidence_from_scores(
            output_dict.sequences, 
            output_dict.scores, 
            input_len,
            processor.tokenizer.pad_token_id
        )

        for j, (path, item, output_text) in enumerate(zip(batch_paths, batch_items, outputs)):
            metrics = item.get("metrics", item) if isinstance(item, dict) else item
            cleaned = output_text.strip()

            # 【关键改动 3】大括号暴力截取法，彻底屏蔽乱码
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx+1]
            else:
                cleaned = "{}"

            try:
                vlm_json = json.loads(cleaned)
                
                # 【关键改动 4】将算出的真实置信度强行覆盖模型复读的 0.9
                true_conf = vlm_confidences[j]
                vlm_json["vlm_confidence"] = round(true_conf, 4)
                
                decision = get_hybrid_decision(vlm_json, metrics, raw_json_failed=False)
                print(f"✅ 成功：{os.path.basename(path)} | 真实置信度: {true_conf:.4f} | 最终决策源: {decision['final_decision_source']}")
            except Exception:
                print(f"❌ JSON 解析失败 → {os.path.basename(path)}")
                decision = get_hybrid_decision({}, metrics, raw_json_failed=True)

            results.append({
                "image_path": path,
                "metrics": metrics,
                "hybrid_evaluation": decision,
                "processing_time": 0.0
            })
    return results

# ==================== 主函数 ====================
def run_pipeline(metrics_json_path: str = "DUO_step1.json",
                 output_json_path: str = "DUO_step2.json",
                 batch_size: int = 8,
                 max_images: int = None):
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_list = json.load(f)
    if max_images:
        metrics_list = metrics_list[:max_images]

    results = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 已有 {len(results)} 条结果，继续处理...")
        except:
            pass

    processed_paths = {item.get("image_path") for item in results if isinstance(item, dict)}
    pending = [item for item in metrics_list if item.get("image_path") not in processed_paths]

    print(f"🚀 开始批量处理剩余 {len(pending)} 张图像 | batch_size={batch_size}")

    for i in tqdm(range(0, len(pending), batch_size), desc="混合代理评估中"):
        batch_items = pending[i:i + batch_size]
        batch_paths = [item["image_path"] for item in batch_items]

        start_time = time.time()
        batch_results = evaluate_batch(batch_paths, batch_items, batch_size)

        for res in batch_results:
            if res:
                res["processing_time"] = round(time.time() - start_time, 2) / max(len(batch_results), 1)
                results.append(res)

        if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= len(pending):
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成！共处理 {len(results)} 张图像")


if __name__ == "__main__":
    run_pipeline(batch_size=8)
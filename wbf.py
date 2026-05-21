import json
from collections import defaultdict

# ================== 配置区 ==================
JSON_FILES = {
    "original": "/root/autodl-tmp/bi/ROUD5_step4_blur.json",
    "dtiuie":   "/root/autodl-tmp/bi/ROUD5_step4_blur.json",
    "dgunet":   "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json"
}

# SOURCE_WEIGHTS = {
#     "original": 1.30,
#     "dtiuie":   1.05,
#     "dgunet":   1.00
# }

SOURCE_WEIGHTS = {
    "original": 1.00,
    "dtiuie":   1.40,   # Blur 下 DTIUIE 最强
    "dgunet":   1.10
}

IOU_THR = 0.9
# ===========================================

data = {}
for name, path in JSON_FILES.items():
    with open(path, 'r') as f:
        data[name] = json.load(f)

img_info = {img["id"]: (img.get("width"), img.get("height")) 
            for img in data["original"].get("images", [])}

image_dets = defaultdict(list)

for name, js in data.items():
    weight = SOURCE_WEIGHTS[name]
    for ann in js.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_info:
            continue
            
        x, y, w, h = ann["bbox"]
        width, height = img_info[img_id]
        
        x1 = max(0.0, min(1.0, x / width))
        y1 = max(0.0, min(1.0, y / height))
        x2 = max(0.0, min(1.0, (x + w) / width))
        y2 = max(0.0, min(1.0, (y + h) / height))
        
        score = float(ann.get("score", 0.95)) * weight
        
        image_dets[img_id].append({
            "box": [x1, y1, x2, y2],
            "score": score,
            "label": int(ann["category_id"])
        })

# 融合（使用最简单的加权 NMS）
fused_annotations = []
ann_id = 1

for img_id, dets in image_dets.items():
    if len(dets) == 0:
        continue
    
    # 按 score 从高到低排序
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    
    boxes = [d["box"] for d in dets]
    scores = [d["score"] for d in dets]
    labels = [d["label"] for d in dets]
    
    # 手动实现简单加权 NMS（避免 ensemble_boxes 的各种 bug）
    keep = []
    for i in range(len(boxes)):
        if i in keep:
            continue
        keep.append(i)
        
        for j in range(i + 1, len(boxes)):
            if j in keep:
                continue
                
            # 计算 IoU
            b1 = boxes[i]
            b2 = boxes[j]
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            
            if iou > IOU_THR:
                # 抑制较低分的框
                if scores[j] < scores[i]:
                    keep.append(j)   # 标记为已抑制（不加入 keep）
    
    # 保存保留的框
    width, height = img_info[img_id]
    for i in keep:
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        
        x1, y1, x2, y2 = box
        x = round(x1 * width, 2)
        y = round(y1 * height, 2)
        w = round((x2 - x1) * width, 2)
        h = round((y2 - y1) * height, 2)
        
        fused_annotations.append({
            "id": ann_id,
            "image_id": int(img_id),
            "category_id": int(label),
            "bbox": [x, y, w, h],
            "score": round(float(score), 4),
            "area": round(w * h, 2),
            "iscrowd": 0
        })
        ann_id += 1

# 保存结果
fused_coco = {
    "images": data["original"]["images"],
    "annotations": fused_annotations,
    "categories": data["original"]["categories"]
}

output_path = "/root/autodl-tmp/bi/ROUD5_step4_blur_wbf2.json"
with open(output_path, "w") as f:
    json.dump(fused_coco, f, indent=2)

print(f"✅ 融合完成！")
print(f"原始总标注数: {sum(len(js.get('annotations', [])) for js in data.values())}")
print(f"融合后标注数: {len(fused_annotations)}")
print(f"结果保存至: {output_path}")
import json
from collections import defaultdict
import numpy as np

# ================== 配置区 ==================
# JSON_FILES = {
#     "original": "/root/autodl-tmp/bi/ROUD5_step4_blur.json",
#     "dtiuie":   "/root/autodl-tmp/bi/ROUD5_step4_blurwE.json",
#     "dgunet":   "/root/autodl-tmp/bi/ROUD5_step4_blurwEdgunet.json"
# }
JSON_FILES = {
    "original": "/root/autodl-tmp/bi/ROUD5_step4_light.json",
    "dtiuie":   "/root/autodl-tmp/bi/ROUD5_step4_lightwE.json",
    "dgunet":   "/root/autodl-tmp/bi/ROUD5_step4_lightwEdgunet.json"
}

# 权重（根据 Light 场景调整）
WEIGHTS = {
    "original": 1.35,
    "dtiuie":   1.05,
    "dgunet":   1.00
}
# WEIGHTS = {
#     "original": 1.00,
#     "dtiuie":   1.40,   # Blur 下 DTIUIE 最强
#     "dgunet":   1.10
# }

IOU_THR = 0.55          # 聚类阈值，0.45~0.6 之间
# ===========================================

# ====================== 辅助函数 ======================
def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)
data = {}
for name, path in JSON_FILES.items():
    with open(path, 'r') as f:
        data[name] = json.load(f)

img_info = {img["id"]: (img.get("width"), img.get("height")) 
            for img in data["original"].get("images", [])}

image_dets = defaultdict(list)

for name, js in data.items():
    wgt = WEIGHTS[name]
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
        
        image_dets[img_id].append({
            "box": np.array([x1, y1, x2, y2]),
            "score": float(ann.get("score", 0.95)) * wgt,
            "label": int(ann["category_id"]),
            "source": name
        })

# 自定义 WBF（聚类 + 加权平均）
fused_annotations = []
ann_id = 1

for img_id, dets in image_dets.items():
    if not dets:
        continue
    
    # 按 score 从高到低排序
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    
    used = [False] * len(dets)
    width, height = img_info[img_id]
    
    for i in range(len(dets)):
        if used[i]:
            continue
        used[i] = True
        
        cluster = [dets[i]]
        
        # 把 IoU > 阈值的框都聚到同一个 cluster
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            iou = _iou(cluster[0]["box"], dets[j]["box"])
            if iou > IOU_THR:
                cluster.append(dets[j])
                used[j] = True
        
        # 对 cluster 做加权平均
        if len(cluster) == 1:
            box = cluster[0]["box"]
            score = cluster[0]["score"]
        else:
            total_score = sum(d["score"] for d in cluster)
            box = np.zeros(4)
            for d in cluster:
                box += d["box"] * d["score"]
            box /= total_score
            score = total_score / len(cluster)   # 简单平均置信度
        
        label = cluster[0]["label"]   # 同一个 cluster 标签通常一致
        
        # 转回像素坐标
        x1, y1, x2, y2 = box
        x = round(x1 * width, 2)
        y = round(y1 * height, 2)
        w = round((x2 - x1) * width, 2)
        h = round((y2 - y1) * height, 2)
        
        fused_annotations.append({
            "id": ann_id,
            "image_id": int(img_id),
            "category_id": label,
            "bbox": [x, y, w, h],
            "score": round(float(score), 4),
            "area": round(w * h, 2),
            "iscrowd": 0
        })
        ann_id += 1

# 保存
fused_coco = {
    "images": data["original"]["images"],
    "annotations": fused_annotations,
    "categories": data["original"]["categories"]
}

output_path = "/root/autodl-tmp/bi/ROUD5_step4_light_wbf2_55.json"
with open(output_path, "w") as f:
    json.dump(fused_coco, f, indent=2)

print(f"✅ 自定义 WBF 融合完成！")
print(f"原始总标注数: {sum(len(js.get('annotations', [])) for js in data.values())}")
print(f"融合后标注数: {len(fused_annotations)}")
print(f"结果保存至: {output_path}")


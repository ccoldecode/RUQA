import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_with_filename_alignment(gt_path, res_path):
    # 1. 加载真值数据 (Ground Truth)
    print(f"📂 正在加载真值表: {gt_path}")
    coco_gt = COCO(gt_path)
    
    # 建立【文件名 -> 真值ID】的映射
    filename_to_gt_id = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}

    # 2. 加载你的推理结果 (Results)
    print(f"🚀 正在加载推理结果并对齐文件名: {res_path}")
    with open(res_path, 'r') as f:
        res_data = json.load(f)
    
    # 建立【你结果里的旧ID -> 文件名】的映射
    # 这样我们就能知道你每一条标注到底对应哪张图
    old_id_to_filename = {img['id']: img['file_name'] for img in res_data['images']}

    aligned_results = []
    skipped_images = 0

    # 3. 核心对齐逻辑
    for ann in res_data['annotations']:
        file_name = old_id_to_filename.get(ann['image_id'])
        
        # 根据文件名寻找它在真值表里的真正 ID
        correct_image_id = filename_to_gt_id.get(file_name)
        
        if correct_image_id is not None:
            # 复制一份标注，更新为正确的 ID
            new_ann = ann.copy()
            new_ann['image_id'] = correct_image_id
            # pycocotools 要求结果中必须有 score，如果没有则默认为 0.95
            if 'score' not in new_ann:
                new_ann['score'] = 0.95
            aligned_results.append(new_ann)
        else:
            skipped_images += 1

    if not aligned_results:
        print("❌ 对齐失败：未能通过文件名匹配到任何标注，请检查文件名是否一致。")
        return

    print(f"✅ 对齐完成：共匹配 {len(aligned_results)} 条标注，跳过 {skipped_images} 条无法匹配的图像。")

    # 4. 使用 pycocotools 进行评测
    # 加载对齐后的结果
    coco_dt = coco_gt.loadRes(aligned_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 打印核心 mAP
    print("\n" + "="*30)
    print(f"📊 {os.path.basename(res_path)} 评测指标:")
    print(f"🔹 mAP @ 0.5: {coco_eval.stats[1]:.4f}")
    print(f"🔹 mAP @ 0.5:0.95: {coco_eval.stats[0]:.4f}")
    print("="*30)

if __name__ == "__main__":
    # 请根据你的 AutoDL 路径修改以下变量
    GT_PATH = "/root/autodl-tmp/bi/results/instances_blur.json"
    RES_PATH = "/root/autodl-tmp/bi/results/step4_instances_blur.json"
    
    evaluate_with_filename_alignment(GT_PATH, RES_PATH)
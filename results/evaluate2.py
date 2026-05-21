import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def final_bulletproof_eval(gt_path, res_path):
    print(f"📂 正在加载真值表: {gt_path}")
    coco_gt = COCO(gt_path)
    
    # 1. 建立【文件名 -> 真值图片 ID】映射
    filename_to_gt_imgid = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}
    
    # 2. 建立【类别名称 -> 真值类别 ID】映射 (极其关键！)
    gt_name_to_catid = {cat['name'].lower(): cat['id'] for cat in coco_gt.dataset['categories']}

    print(f"🚀 正在加载推理结果并进行双重对齐: {res_path}")
    with open(res_path, 'r') as f:
        res_data = json.load(f)
    
    # 3. 建立【预测图片旧 ID -> 文件名】映射
    old_imgid_to_filename = {img['id']: img['file_name'] for img in res_data['images']}
    
    # 4. 你在 step4 脚本中定义的类别映射 (写死在这里，防止错乱)
    MY_ID_TO_NAME = {
        1: "holothurian", 2: "echinus", 3: "scallop", 4: "starfish", 
        5: "fish", 6: "corals", 7: "diver", 8: "cuttlefish", 9: "turtle", 10: "jellyfish"
    }

    aligned_results = []
    skipped_boxes = 0

    # ================= 核心：逐个框进行双重对齐 =================
    # 无论一张图片有几个框，这里都会把每个框的图片 ID 和 类别 ID 都纠正过来
    for idx, ann in enumerate(res_data['annotations']):
        # 获取该框对应的文件名
        file_name = old_imgid_to_filename.get(ann['image_id'])
        # 根据文件名找正确的图片 ID
        correct_img_id = filename_to_gt_imgid.get(file_name)
        
        # 获取该框预测出的名字
        pred_cat_name = MY_ID_TO_NAME.get(ann['category_id'], "").lower()
        # 根据名字找正确的类别 ID
        correct_cat_id = gt_name_to_catid.get(pred_cat_name)
        
        if correct_img_id and correct_cat_id:
            new_ann = ann.copy()
            # 重新分配一个绝对不重复的标注 ID（防止多框 ID 冲突）
            new_ann['id'] = idx + 1  
            # 修正图片 ID
            new_ann['image_id'] = correct_img_id
            # 修正类别 ID
            new_ann['category_id'] = correct_cat_id
            
            # 补充置信度
            if 'score' not in new_ann:
                new_ann['score'] = 0.95
                
            aligned_results.append(new_ann)
        else:
            skipped_boxes += 1

    if not aligned_results:
        print("❌ 灾难性失败：没有一个框对齐成功，请检查上方映射逻辑。")
        return

    print(f"✅ 对齐完成！")
    print(f"   - 成功重组并对齐标注框: {len(aligned_results)} 个")
    print(f"   - 因无法匹配被丢弃的框: {skipped_boxes} 个")

    # ================= 运行 COCO 评测 =================
    coco_dt = coco_gt.loadRes(aligned_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n" + "="*30)
    print(f"📊 最终真实 mAP ({os.path.basename(res_path)}):")
    print(f"🔹 mAP @ 0.5: {coco_eval.stats[1]:.4f} (这就是你论文里要写的分数)")
    print(f"🔹 mAP @ 0.5:0.95: {coco_eval.stats[0]:.4f}")
    print("="*30)

if __name__ == "__main__":
    GT_PATH = "/root/autodl-tmp/bi/results/instances_blur.json"
    RES_PATH = "/root/autodl-tmp/bi/ROUD_step4_blur.json"
    
    final_bulletproof_eval(GT_PATH, RES_PATH)
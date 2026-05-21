import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def final_bulletproof_eval(gt_path, res_path):
    print(f"\n" + "="*50)
    print(f"📂 正在加载真值表: {gt_path}")
    coco_gt = COCO(gt_path)
    
    # 1. 建立【文件名 -> 真值图片 ID】映射
    filename_to_gt_imgid = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}
    
    # 2. 建立【类别名称 -> 真值类别 ID】映射
    gt_name_to_catid = {cat['name'].lower(): cat['id'] for cat in coco_gt.dataset['categories']}

    print(f"🚀 正在加载推理结果并进行双重对齐: {res_path}")
    with open(res_path, 'r') as f:
        res_data = json.load(f)
    
    # 3. 建立【预测图片旧 ID -> 文件名】映射
    old_imgid_to_filename = {img['id']: img['file_name'] for img in res_data['images']}
    
    # 4. 类别映射
    MY_ID_TO_NAME = {
        1: "holothurian", 2: "echinus", 3: "scallop", 4: "starfish", 
        5: "fish", 6: "corals", 7: "diver", 8: "cuttlefish", 9: "turtle", 10: "jellyfish"
    }
    # # 4. 类别映射 (将预测的 category_id 直接映射为 GT 里的学名)
    # MY_ID_TO_NAME = {
    #     1: "holothurian",   # 对应预测出的 sea cucumber
    #     2: "echinus",       # 对应预测出的 sea urchin
    #     3: "scallop",       # 对应预测出的 scallop
    #     4: "starfish",      # 对应预测出的 starfish
    #     7: "fish",          # 你的 fish 对应的是 ID 7
    #     8: "jellyfish",     # 你的 jellyfish 对应的是 ID 8
    #     9: "turtle",        # 你的 turtle 对应的是 ID 9
    #     10: "cuttlefish",   # 对应预测出的 squid
    #     11: "corals",       # 你的 corals 对应的是 ID 11
    #     14: "diver",        # 你的 diver 对应的是 ID 14
    #     16: "marine debris" # 对应预测出的海洋垃圾 (如果 GT 有对应的话)
    # }

    aligned_results = []
    skipped_boxes = 0

    # 对齐逻辑
    for idx, ann in enumerate(res_data['annotations']):
        file_name = old_imgid_to_filename.get(ann['image_id'])
        correct_img_id = filename_to_gt_imgid.get(file_name)
        
        pred_cat_name = MY_ID_TO_NAME.get(ann['category_id'], "").lower()
        correct_cat_id = gt_name_to_catid.get(pred_cat_name)
        
        if correct_img_id and correct_cat_id:
            new_ann = ann.copy()
            new_ann['id'] = idx + 1  
            new_ann['image_id'] = correct_img_id
            new_ann['category_id'] = correct_cat_id
            if 'score' not in new_ann:
                new_ann['score'] = 0.95
            aligned_results.append(new_ann)
        else:
            skipped_boxes += 1

    if not aligned_results:
        print(f"❌ 对齐失败: {res_path}")
        return None

    # 运行评测
    coco_dt = coco_gt.loadRes(aligned_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回 mAP @ 0.5
    return coco_eval.stats[1]

if __name__ == "__main__":
    # 基础目录
    BASE_DIR = "/root/autodl-tmp/bi"
    
    # 定义三个子集的路径
    # tasks = [
    #     {"name": "Blur",  "gt": "results/instances_blur.json",  "res": "ROUD2_step4_blurwE_dgunet.json"},
    #     {"name": "Color", "gt": "results/instances_color.json", "res": "ROUD2_step4_colorwE_dgunet.json"},
    #     {"name": "Light", "gt": "results/instances_light.json", "res": "ROUD2_step4_lightwE_dgunet.json"}
    # ]
    # tasks = [
    #     {"name": "Blur",  "gt": "results/instances_blur.json",  "res": "ROUD5_step4_blurwEdgunet.json"},
    #     {"name": "Color", "gt": "results/instances_color.json", "res": "ROUD5_step4_colorwEdgunet.json"},
    #     {"name": "Light", "gt": "results/instances_light.json", "res": "ROUD5_step4_lightwEdgunet.json"}
    # ]
    # tasks = [
    #     {"name": "Blur",  "gt": "results/instances_blur.json",  "res": "ROUD5_step4_blur.json"},
    #     {"name": "Color", "gt": "results/instances_color.json", "res": "ROUD5_step4_color.json"},
    #     {"name": "Light", "gt": "results/instances_light.json", "res": "ROUD5_step4_light.json"}
    # ]
    # tasks = [
    #     {"name": "Blur",  "gt": "results/instances_blur.json",  "res": "ROUD5_step4_blurwE.json"},
    #     {"name": "Color", "gt": "results/instances_color.json", "res": "ROUD5_step4_colorwE.json"},
    #     {"name": "Light", "gt": "results/instances_light.json", "res": "ROUD5_step4_lightwE.json"}
    # ]
    tasks = [
    #     {"name": "DUOWE",  "gt": "instances_DUO.json",  "res": "DUO_step4_wE.json"},
    #     {"name": "DUO",  "gt": "instances_DUO.json",  "res": "DUO_step4.json"}
        {"name": "Blur", "gt": "results/instances_blur.json", "res": "ROUD5_step5_blur3.json"},
        {"name": "Color", "gt": "results/instances_color.json", "res": "ROUD5_step5_color3.json"},
        {"name": "Light", "gt": "results/instances_light.json", "res": "ROUD5_step5_light3.json"}
    ]

    all_scores = {}

    # 开始循环测试
    for task in tasks:
        gt_p = os.path.join(BASE_DIR, task["gt"])
        res_p = os.path.join(BASE_DIR, task["res"])
        
        if os.path.exists(gt_p) and os.path.exists(res_p):
            mAP = final_bulletproof_eval(gt_p, res_p)
            if mAP is not None:
                all_scores[task["name"]] = mAP
        else:
            print(f"⚠️ 跳过 {task['name']}，文件不存在。")

    # ================= 最终结果统计 =================
    print("\n" + "★"*20 + " 毕业论文数据汇总 " + "★"*20)
    for name, score in all_scores.items():
        print(f"🔹 {name:<10} mAP@0.5: {score:.4f}")
        # pass
    
    if all_scores:
        avg_score = sum(all_scores.values()) / len(all_scores)
        print(f"🏆 Overall Robustness (Total mAP): {avg_score:.4f}")
    print("★"*58)
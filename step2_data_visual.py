import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# 修改为系统通常自带的字体名称
FONT_NAME = 'WenQuanYi Micro Hei' 

def setup_academic_font():
    plt.rcParams['font.sans-serif'] = [FONT_NAME, 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"尝试使用字体: {FONT_NAME}")
    return FONT_NAME

# ================= 配置区 =================
# 请确保该文件已上传到当前目录
FONT_PATH = 'SimHei.ttf' 
# =========================================

# def setup_academic_font():
#     """手动加载本地字体文件"""
#     if os.path.exists(FONT_PATH):
#         # 强制注册字体
#         fm.fontManager.addfont(FONT_PATH)
#         prop = fm.FontProperties(fname=FONT_PATH)
#         plt.rcParams['font.sans-serif'] = [prop.get_name()]
#         plt.rcParams['axes.unicode_minus'] = False
#         print(f"✅ 成功加载本地字体: {prop.get_name()}")
#         return prop.get_name()
#     else:
#         print(f"❌ 未找到字体文件 {FONT_PATH}，请确认已上传！")
#         return 'DejaVu Sans'

def analyze_distribution(json_path="all_step2_fixed.json", output_dir="step2_analysis_results0507"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for item in data:
        eval_res = item.get("hybrid_evaluation", {})
        tags = eval_res.get("tags", {})
        scores = eval_res.get("severity_scores", {})
        rows.append({
            "浑浊度等级": tags.get("turbidity"),
            "偏色等级": tags.get("color_cast"),
            "亮度等级": tags.get("brightness"),
            "浑浊度分值": scores.get("turbidity_score", 0),
            "偏色分值": scores.get("color_score", 0),
            "亮度分值": scores.get("brightness_score", 0),
            "组合标签": tags.get("combined_tag")
        })

    df = pd.DataFrame(rows)
    
    # 初始化字体和风格
    font_family = setup_academic_font()
    sns.set_theme(style="ticks", font=font_family)

    # 1. 综合报告图
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"URB 数据集分布特征分析 (总样本数: {len(df)})", fontsize=20, fontweight='bold', y=0.98)

    order_map = {"浑浊度等级": ["L1", "L2", "L3"], 
                 "偏色等级": ["C1", "C2", "C3"], 
                 "亮度等级": ["B1", "B2", "B3"]}
    titles = ["浑浊度等级分布 (L)", "偏色等级分布 (C)", "亮度等级分布 (B)"]
    cols = ["浑浊度等级", "偏色等级", "亮度等级"]

    for i, col in enumerate(cols):
        # 修复 FutureWarning: 显式指定 hue
        sns.countplot(data=df, x=col, ax=axes[0, i], hue=col, palette="mako", order=order_map[col], legend=False)
        axes[0, i].set_title(titles[i], fontsize=15)
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("样本数量")
        for p in axes[0, i].patches:
            axes[0, i].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='baseline', fontsize=11, xytext=(0, 5), textcoords='offset points')

    score_cols = ["浑浊度分值", "偏色分值", "亮度分值"]
    colors = ['#2c7fb8', '#31a354', '#de2d26']
    for i, col in enumerate(score_cols):
        sns.histplot(data=df, x=col, ax=axes[1, i], kde=True, color=colors[i], bins=15, alpha=0.5)
        axes[1, i].set_title(f"{col}密度分布", fontsize=15)
        axes[1, i].set_xlabel("分值 (0-10)")
        axes[1, i].set_ylabel("频数")
        axes[1, i].set_xlim(0, 10)

    sns.despine()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/数据分布报告.png", dpi=300)

    # 2. Top 10 组合标签
    plt.figure(figsize=(14, 7))
    top_tags = df['组合标签'].value_counts().head(10)
    sns.barplot(x=top_tags.index, y=top_tags.values, hue=top_tags.index, palette="rocket", legend=False)
    plt.title("前十类组合退化类型分布", fontsize=18, fontweight='bold')
    plt.xticks(rotation=30)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/组合标签排名.png", dpi=300)

    print(f"\n✅ 分析完成！图像已生成，请检查 '{output_dir}'。")

if __name__ == "__main__":
    analyze_distribution()
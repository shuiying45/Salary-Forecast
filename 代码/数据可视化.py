import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# =========================
# 1. 路径设置
# =========================
base_dir = r"C:\Users\shenjie\Desktop\学校\商务数据管理\案例2"
file_path = os.path.join(base_dir, "处理后_技术岗位薪酬数据.xlsx")
output_dir = os.path.join(base_dir, "图表输出")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =========================
# 2. 中文字体设置
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", rc={
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
    "axes.unicode_minus": False
})

# =========================
# 3. 读取数据
# =========================
df = pd.read_excel(file_path)

print("数据读取成功！")
print("数据维度：", df.shape)

# 为散点图抽样，避免图片太密
plot_df = df.sample(n=30000, random_state=42) if len(df) > 30000 else df.copy()

# =========================
# 4. 薪资分布直方图
# =========================
plt.figure(figsize=(10, 6))
sns.histplot(df["薪资"], bins=50, kde=True)
plt.title("技术岗位薪资分布直方图")
plt.xlabel("薪资")
plt.ylabel("岗位数量")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_薪资分布直方图.png"), dpi=300)
plt.close()

# =========================
# 5. 薪资箱线图
# =========================
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["薪资"])
plt.title("技术岗位薪资箱线图")
plt.xlabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_薪资箱线图.png"), dpi=300)
plt.close()

# =========================
# 6. 不同岗位平均薪资
# =========================
job_salary = df.groupby("岗位名称")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=job_salary.index, y=job_salary.values)
plt.title("不同岗位平均薪资对比")
plt.xlabel("岗位名称")
plt.ylabel("平均薪资")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_不同岗位平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 7. 不同学历平均薪资
# =========================
edu_salary = df.groupby("学历水平")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=edu_salary.index, y=edu_salary.values)
plt.title("不同学历水平平均薪资对比")
plt.xlabel("学历水平")
plt.ylabel("平均薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_不同学历平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 8. 学历与薪资箱线图
# =========================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="学历水平", y="薪资")
plt.title("不同学历水平薪资分布箱线图")
plt.xlabel("学历水平")
plt.ylabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_学历水平与薪资箱线图.png"), dpi=300)
plt.close()

# =========================
# 9. 工作经验与薪资散点图
# =========================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="工作经验年限", y="薪资", alpha=0.35)
plt.title("工作经验年限与薪资关系散点图")
plt.xlabel("工作经验年限")
plt.ylabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_工作经验年限与薪资散点图.png"), dpi=300)
plt.close()

# =========================
# 10. 工作经验与薪资回归趋势图
# =========================
plt.figure(figsize=(10, 6))
sns.regplot(data=plot_df, x="工作经验年限", y="薪资", scatter_kws={"alpha": 0.25}, line_kws={"linewidth": 2})
plt.title("工作经验年限与薪资回归趋势图")
plt.xlabel("工作经验年限")
plt.ylabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "07_工作经验年限与薪资趋势图.png"), dpi=300)
plt.close()

# =========================
# 11. 技能数量与薪资散点图
# =========================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="技能数量", y="薪资", alpha=0.35)
plt.title("技能数量与薪资关系散点图")
plt.xlabel("技能数量")
plt.ylabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "08_技能数量与薪资散点图.png"), dpi=300)
plt.close()

# =========================
# 12. 技能数量与薪资回归趋势图
# =========================
plt.figure(figsize=(10, 6))
sns.regplot(data=plot_df, x="技能数量", y="薪资", scatter_kws={"alpha": 0.25}, line_kws={"linewidth": 2})
plt.title("技能数量与薪资回归趋势图")
plt.xlabel("技能数量")
plt.ylabel("薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "09_技能数量与薪资趋势图.png"), dpi=300)
plt.close()

# =========================
# 13. 公司规模与平均薪资
# =========================
company_salary = df.groupby("公司规模")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=company_salary.index, y=company_salary.values)
plt.title("不同公司规模平均薪资对比")
plt.xlabel("公司规模")
plt.ylabel("平均薪资")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "10_不同公司规模平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 14. 地区与平均薪资
# =========================
location_salary = df.groupby("工作地区")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=location_salary.index, y=location_salary.values)
plt.title("不同工作地区平均薪资对比")
plt.xlabel("工作地区")
plt.ylabel("平均薪资")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "11_不同工作地区平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 15. 所属行业与平均薪资
# =========================
industry_salary = df.groupby("所属行业")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=industry_salary.index, y=industry_salary.values)
plt.title("不同行业平均薪资对比")
plt.xlabel("所属行业")
plt.ylabel("平均薪资")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "12_不同行业平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 16. 远程办公方式与平均薪资
# =========================
remote_salary = df.groupby("远程办公方式")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=remote_salary.index, y=remote_salary.values)
plt.title("不同远程办公方式平均薪资对比")
plt.xlabel("远程办公方式")
plt.ylabel("平均薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "13_远程办公方式与平均薪资.png"), dpi=300)
plt.close()

# =========================
# 17. 薪资区间占比饼图
# =========================
salary_level_count = df["薪资区间"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    salary_level_count,
    labels=salary_level_count.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("薪资区间占比")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "14_薪资区间占比饼图.png"), dpi=300)
plt.close()

# =========================
# 18. 数值变量相关性热力图
# =========================
num_cols = ["工作经验年限", "技能数量", "证书数量", "薪资"]
corr = df[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("数值变量相关性热力图")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "15_相关性热力图.png"), dpi=300)
plt.close()

# =========================
# 19. 经验等级与平均薪资
# =========================
exp_salary = df.groupby("经验等级")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=exp_salary.index, y=exp_salary.values)
plt.title("不同经验等级平均薪资对比")
plt.xlabel("经验等级")
plt.ylabel("平均薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "16_不同经验等级平均薪资对比.png"), dpi=300)
plt.close()

# =========================
# 20. 技能等级与平均薪资
# =========================
skill_salary = df.groupby("技能等级")["薪资"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=skill_salary.index, y=skill_salary.values)
plt.title("不同技能等级平均薪资对比")
plt.xlabel("技能等级")
plt.ylabel("平均薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "17_不同技能等级平均薪资对比.png"), dpi=300)
plt.close()

print("数据探索可视化完成！")
print("图表保存位置：", output_dir)
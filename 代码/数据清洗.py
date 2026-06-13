import pandas as pd
import os

# =========================
# 1. 路径设置
# =========================
input_path = r"C:\Users\shenjie\Desktop\学校\商务数据管理\job_salary_prediction_dataset.xlsx"
base_dir = r"C:\Users\shenjie\Desktop\学校\商务数据管理\案例2"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

output_path = os.path.join(base_dir, "处理后_技术岗位薪酬数据.xlsx")

# =========================
# 2. 读取数据
# =========================
df = pd.read_excel(input_path)

print("数据读取成功！")
print("原始数据维度：", df.shape)
print("\n原始字段：")
print(df.columns)

# =========================
# 3. 字段中文化
# =========================
df = df.rename(columns={
    "job_title": "岗位名称",
    "experience_years": "工作经验年限",
    "education_level": "学历水平",
    "skills_count": "技能数量",
    "industry": "所属行业",
    "company_size": "公司规模",
    "location": "工作地区",
    "remote_work": "远程办公方式",
    "certifications": "证书数量",
    "salary": "薪资"
})

print("\n中文字段：")
print(df.columns)

# =========================
# 4. 缺失值检查与处理
# =========================
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["字段", "缺失值数量"]
missing_df["缺失比例"] = missing_df["缺失值数量"] / len(df)

print("\n缺失值统计：")
print(missing_df)

df = df.dropna()

# =========================
# 5. 重复值检查与处理
# =========================
duplicate_count = df.duplicated().sum()
print("\n重复值数量：", duplicate_count)

df = df.drop_duplicates()

# =========================
# 6. 薪资异常值处理
# =========================
df = df[df["薪资"] > 0]

print("\n薪资描述性统计：")
print(df["薪资"].describe())

# =========================
# 7. 构造薪资区间
# =========================
def salary_level(salary):
    if salary < 100000:
        return "低薪区间"
    elif salary < 150000:
        return "中低薪区间"
    elif salary < 200000:
        return "中高薪区间"
    else:
        return "高薪区间"

df["薪资区间"] = df["薪资"].apply(salary_level)

# =========================
# 8. 构造经验等级
# =========================
def experience_level(years):
    if years <= 2:
        return "初级经验"
    elif years <= 5:
        return "中级经验"
    elif years <= 10:
        return "高级经验"
    else:
        return "资深经验"

df["经验等级"] = df["工作经验年限"].apply(experience_level)

# =========================
# 9. 构造技能等级
# =========================
def skill_level(count):
    if count <= 3:
        return "技能较少"
    elif count <= 6:
        return "技能中等"
    else:
        return "技能较多"

df["技能等级"] = df["技能数量"].apply(skill_level)

# =========================
# 10. 保存数据质量统计
# =========================
quality_path = os.path.join(base_dir, "数据质量统计.xlsx")

salary_level_count = df["薪资区间"].value_counts().reset_index()
salary_level_count.columns = ["薪资区间", "数量"]

exp_level_count = df["经验等级"].value_counts().reset_index()
exp_level_count.columns = ["经验等级", "数量"]

skill_level_count = df["技能等级"].value_counts().reset_index()
skill_level_count.columns = ["技能等级", "数量"]

with pd.ExcelWriter(quality_path) as writer:
    missing_df.to_excel(writer, sheet_name="缺失值统计", index=False)
    salary_level_count.to_excel(writer, sheet_name="薪资区间分布", index=False)
    exp_level_count.to_excel(writer, sheet_name="经验等级分布", index=False)
    skill_level_count.to_excel(writer, sheet_name="技能等级分布", index=False)

# =========================
# 11. 保存处理后数据
# =========================
df.to_excel(output_path, index=False)

print("\n数据预处理完成！")
print("处理后数据维度：", df.shape)
print("处理后数据保存位置：", output_path)
print("数据质量统计保存位置：", quality_path)
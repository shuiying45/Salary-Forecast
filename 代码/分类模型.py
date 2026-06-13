import pandas as pd
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# =========================
# 1. 路径设置
# =========================
base_dir = r"C:\Users\shenjie\Desktop\学校\商务数据管理\案例2"
file_path = os.path.join(base_dir, "处理后_技术岗位薪酬数据.xlsx")
output_dir = os.path.join(base_dir, "图表输出")
model_dir = os.path.join(base_dir, "模型结果")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

result_output_path = os.path.join(model_dir, "分类模型评价结果.xlsx")
confusion_output_path = os.path.join(model_dir, "分类模型混淆矩阵.xlsx")
report_output_path = os.path.join(model_dir, "分类模型详细报告.xlsx")
prediction_output_path = os.path.join(model_dir, "最优分类模型预测明细.xlsx")

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

# =========================
# 4. 特征与目标变量
# =========================
feature_cols = [
    "岗位名称",
    "工作经验年限",
    "学历水平",
    "技能数量",
    "所属行业",
    "公司规模",
    "工作地区",
    "远程办公方式",
    "证书数量"
]

target_col = "薪资区间"

X = df[feature_cols]
y = df[target_col]

print("\n薪资区间分布：")
print(y.value_counts())

categorical_features = [
    "岗位名称",
    "学历水平",
    "所属行业",
    "公司规模",
    "工作地区",
    "远程办公方式"
]

numeric_features = [
    "工作经验年限",
    "技能数量",
    "证书数量"
]

# =========================
# 5. OneHotEncoder兼容写法
# =========================
try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("类别变量", onehot, categorical_features),
        ("数值变量", StandardScaler(), numeric_features)
    ]
)

# =========================
# 6. 划分完整训练集和测试集
# =========================
class_labels = ["低薪区间", "中低薪区间", "中高薪区间", "高薪区间"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7. 抽样数据，用于KNN
# =========================
sample_size = 30000

if len(df) > sample_size:
    df_sample = df.sample(n=sample_size, random_state=42)
else:
    df_sample = df.copy()

X_sample = df_sample[feature_cols]
y_sample = df_sample[target_col]

X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample
)

print("抽样数据维度：", df_sample.shape)

# =========================
# 8. 定义分类模型
# =========================
models = {
    "Logistic回归": {
        "model": LogisticRegression(max_iter=1000),
        "use_sample": False
    },
    "KNN分类": {
        "model": KNeighborsClassifier(n_neighbors=5),
        "use_sample": True
    },
    "决策树分类": {
        "model": DecisionTreeClassifier(
            random_state=42,
            max_depth=12
        ),
        "use_sample": False
    },
    "随机森林分类": {
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        ),
        "use_sample": False
    },
    "梯度提升分类": {
        "model": GradientBoostingClassifier(
            random_state=42,
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3
        ),
        "use_sample": False
    }
}

# =========================
# 9. 模型训练与评价
# =========================
results = []
confusion_matrices = {}
classification_reports = {}
trained_pipelines = {}

best_model_name = None
best_f1 = -1
best_y_true = None
best_y_pred = None
best_X_test = None

for name, item in models.items():
    print("\n正在训练模型：", name)

    model = item["model"]
    use_sample = item["use_sample"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if use_sample:
        pipeline.fit(X_train_sample, y_train_sample)
        y_pred = pipeline.predict(X_test_sample)
        y_true = y_test_sample
        X_eval = X_test_sample
        data_note = "抽样数据"
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_true = y_test
        X_eval = X_test
        data_note = "完整数据"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    results.append({
        "模型": name,
        "使用数据": data_note,
        "Accuracy准确率": round(accuracy, 4),
        "Precision精确率": round(precision, 4),
        "Recall召回率": round(recall, 4),
        "F1-score": round(f1, 4)
    })

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"真实_{label}" for label in class_labels],
        columns=[f"预测_{label}" for label in class_labels]
    )
    confusion_matrices[name] = cm_df

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).T
    classification_reports[name] = report_df

    trained_pipelines[name] = pipeline

    print(f"{name} 完成：Accuracy={accuracy:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_y_true = y_true
        best_y_pred = y_pred
        best_X_test = X_eval

# =========================
# 10. 保存评价结果
# =========================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="F1-score", ascending=False)

result_df.to_excel(result_output_path, index=False)

with pd.ExcelWriter(confusion_output_path) as writer:
    for name, cm_df in confusion_matrices.items():
        cm_df.to_excel(writer, sheet_name=name[:31])

with pd.ExcelWriter(report_output_path) as writer:
    for name, report_df in classification_reports.items():
        report_df.to_excel(writer, sheet_name=name[:31])

best_pred_df = pd.DataFrame({
    "真实薪资区间": best_y_true.values,
    "预测薪资区间": best_y_pred
})
best_pred_df["是否预测正确"] = best_pred_df["真实薪资区间"] == best_pred_df["预测薪资区间"]
best_pred_df.to_excel(prediction_output_path, index=False)

print("\n分类模型评价结果已保存：", result_output_path)
print("分类模型混淆矩阵已保存：", confusion_output_path)
print("分类模型详细报告已保存：", report_output_path)
print("最优分类模型：", best_model_name)
print("最优分类模型预测明细已保存：", prediction_output_path)

# =========================
# 11. Accuracy对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="Accuracy准确率")
plt.title("不同分类模型Accuracy准确率对比")
plt.xlabel("分类模型")
plt.ylabel("Accuracy准确率")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "28_分类模型Accuracy对比.png"), dpi=300)
plt.close()

# =========================
# 12. Precision对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="Precision精确率")
plt.title("不同分类模型Precision精确率对比")
plt.xlabel("分类模型")
plt.ylabel("Precision精确率")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "29_分类模型Precision对比.png"), dpi=300)
plt.close()

# =========================
# 13. Recall对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="Recall召回率")
plt.title("不同分类模型Recall召回率对比")
plt.xlabel("分类模型")
plt.ylabel("Recall召回率")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "30_分类模型Recall对比.png"), dpi=300)
plt.close()

# =========================
# 14. F1-score对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="F1-score")
plt.title("不同分类模型F1-score对比")
plt.xlabel("分类模型")
plt.ylabel("F1-score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "31_分类模型F1-score对比.png"), dpi=300)
plt.close()

# =========================
# 15. 分类模型指标热力图
# =========================
metric_df = result_df.set_index("模型")[[
    "Accuracy准确率",
    "Precision精确率",
    "Recall召回率",
    "F1-score"
]]

plt.figure(figsize=(10, 6))
sns.heatmap(metric_df, annot=True, fmt=".4f", cmap="Blues")
plt.title("分类模型评价指标热力图")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "32_分类模型评价指标热力图.png"), dpi=300)
plt.close()

# =========================
# 16. 所有模型混淆矩阵
# =========================
for name, cm_df in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name}混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    safe_name = name.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(output_dir, f"33_{safe_name}_混淆矩阵.png"), dpi=300)
    plt.close()

# =========================
# 17. 最优模型混淆矩阵
# =========================
best_cm = confusion_matrices[best_model_name]

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{best_model_name}最佳分类模型混淆矩阵")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "34_最佳分类模型混淆矩阵.png"), dpi=300)
plt.close()

# =========================
# 18. 真实类别与预测类别分布对比
# =========================
compare_df = pd.DataFrame({
    "真实类别": best_y_true.values,
    "预测类别": best_y_pred
})

true_count = compare_df["真实类别"].value_counts().reindex(class_labels).fillna(0)
pred_count = compare_df["预测类别"].value_counts().reindex(class_labels).fillna(0)

count_compare_df = pd.DataFrame({
    "薪资区间": class_labels,
    "真实数量": true_count.values,
    "预测数量": pred_count.values
})

count_compare_long = count_compare_df.melt(
    id_vars="薪资区间",
    value_vars=["真实数量", "预测数量"],
    var_name="类别",
    value_name="数量"
)

plt.figure(figsize=(10, 6))
sns.barplot(data=count_compare_long, x="薪资区间", y="数量", hue="类别")
plt.title(f"{best_model_name}真实类别与预测类别数量对比")
plt.xlabel("薪资区间")
plt.ylabel("数量")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "35_最佳分类模型真实预测类别分布对比.png"), dpi=300)
plt.close()

# =========================
# 19. 预测正确与错误占比
# =========================
correct_count = best_pred_df["是否预测正确"].value_counts()
correct_count.index = ["预测正确" if x else "预测错误" for x in correct_count.index]

plt.figure(figsize=(8, 8))
plt.pie(
    correct_count,
    labels=correct_count.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title(f"{best_model_name}预测正确与错误占比")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "36_最佳分类模型预测正确错误占比.png"), dpi=300)
plt.close()

# =========================
# 20. 分类模型特征重要性
# =========================
importance_model_name = None

for candidate in ["随机森林分类", "梯度提升分类", "决策树分类"]:
    if candidate in trained_pipelines:
        importance_model_name = candidate
        break

if importance_model_name is not None:
    importance_pipeline = trained_pipelines[importance_model_name]
    clf_model = importance_pipeline.named_steps["model"]

    try:
        feature_names = importance_pipeline.named_steps["preprocessor"].get_feature_names_out()
        feature_names = [name.replace("类别变量__", "").replace("数值变量__", "") for name in feature_names]

        if hasattr(clf_model, "feature_importances_"):
            importances = clf_model.feature_importances_

            importance_df = pd.DataFrame({
                "特征": feature_names,
                "重要性": importances
            }).sort_values(by="重要性", ascending=False).head(20)

            importance_df.to_excel(os.path.join(model_dir, "分类模型特征重要性.xlsx"), index=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x="重要性", y="特征")
            plt.title(f"{importance_model_name}特征重要性Top20")
            plt.xlabel("重要性")
            plt.ylabel("特征")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "37_分类模型特征重要性Top20.png"), dpi=300)
            plt.close()

    except Exception as e:
        print("分类模型特征重要性图生成失败：", e)

print("分类模型增强版运行完成！")
print("图表保存位置：", output_dir)
print("模型结果保存位置：", model_dir)
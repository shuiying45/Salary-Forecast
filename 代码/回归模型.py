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

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

result_output_path = os.path.join(model_dir, "回归模型评价结果.xlsx")
prediction_output_path = os.path.join(model_dir, "最优回归模型预测明细.xlsx")

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
# 3. 尝试导入 XGBoost
# =========================
try:
    from xgboost import XGBRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("当前环境未安装 xgboost，将跳过 XGBoost 回归模型。")
    print("安装命令：pip install xgboost")

# =========================
# 4. 读取数据
# =========================
df = pd.read_excel(file_path)

print("数据读取成功！")
print("数据维度：", df.shape)

# =========================
# 5. 特征与目标变量
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

target_col = "薪资"

X = df[feature_cols]
y = df[target_col]

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
# 6. OneHotEncoder兼容写法
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
# 7. 划分完整训练集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 8. 抽样数据，用于KNN和SVR
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
    random_state=42
)

print("抽样数据维度：", df_sample.shape)

# =========================
# 9. 定义回归模型
# =========================
models = {
    "线性回归": {
        "model": LinearRegression(),
        "use_sample": False
    },
    "岭回归": {
        "model": Ridge(alpha=1.0),
        "use_sample": False
    },
    "KNN回归": {
        "model": KNeighborsRegressor(n_neighbors=5),
        "use_sample": True
    },
    "SVR支持向量回归": {
        "model": SVR(kernel="rbf", C=100, gamma="scale"),
        "use_sample": True
    },
    "决策树回归": {
        "model": DecisionTreeRegressor(
            random_state=42,
            max_depth=12
        ),
        "use_sample": False
    },
    "随机森林回归": {
        "model": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        ),
        "use_sample": False
    },
    "梯度提升回归": {
        "model": GradientBoostingRegressor(
            random_state=42,
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3
        ),
        "use_sample": False
    }
}

if xgboost_available:
    models["XGBoost回归"] = {
        "model": XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1
        ),
        "use_sample": False
    }

# =========================
# 10. 模型训练与评价
# =========================
results = []
prediction_records = {}
trained_pipelines = {}

best_model_name = None
best_rmse = float("inf")
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

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    results.append({
        "模型": name,
        "使用数据": data_note,
        "MAE平均绝对误差": round(mae, 4),
        "MSE均方误差": round(mse, 4),
        "RMSE均方根误差": round(rmse, 4),
        "R2拟合优度": round(r2, 4)
    })

    prediction_records[name] = pd.DataFrame({
        "真实薪资": y_true.values,
        "预测薪资": y_pred,
        "预测误差": y_true.values - y_pred,
        "绝对误差": np.abs(y_true.values - y_pred)
    })

    trained_pipelines[name] = pipeline

    print(f"{name} 完成：MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_y_true = y_true
        best_y_pred = y_pred
        best_X_test = X_eval

# =========================
# 11. 保存评价结果
# =========================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="RMSE均方根误差", ascending=True)

result_df.to_excel(result_output_path, index=False)

best_pred_df = prediction_records[best_model_name]
best_pred_df.to_excel(prediction_output_path, index=False)

print("\n回归模型评价结果已保存：", result_output_path)
print("最优回归模型：", best_model_name)
print("最优模型预测明细已保存：", prediction_output_path)

# =========================
# 12. MAE对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="MAE平均绝对误差")
plt.title("不同回归模型MAE对比")
plt.xlabel("回归模型")
plt.ylabel("MAE平均绝对误差")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "18_回归模型MAE对比.png"), dpi=300)
plt.close()

# =========================
# 13. RMSE对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="RMSE均方根误差")
plt.title("不同回归模型RMSE对比")
plt.xlabel("回归模型")
plt.ylabel("RMSE均方根误差")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "19_回归模型RMSE对比.png"), dpi=300)
plt.close()

# =========================
# 14. R2对比图
# =========================
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x="模型", y="R2拟合优度")
plt.title("不同回归模型R2对比")
plt.xlabel("回归模型")
plt.ylabel("R2拟合优度")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "20_回归模型R2对比.png"), dpi=300)
plt.close()

# =========================
# 15. 回归指标热力图
# =========================
metric_df = result_df.set_index("模型")[[
    "MAE平均绝对误差",
    "RMSE均方根误差",
    "R2拟合优度"
]]

plt.figure(figsize=(10, 6))
sns.heatmap(metric_df, annot=True, fmt=".4f", cmap="Blues")
plt.title("回归模型评价指标热力图")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "21_回归模型评价指标热力图.png"), dpi=300)
plt.close()

# =========================
# 16. 最优模型真实值 vs 预测值
# =========================
plot_n = min(8000, len(best_pred_df))
plot_best = best_pred_df.sample(n=plot_n, random_state=42)

plt.figure(figsize=(8, 8))
sns.scatterplot(data=plot_best, x="真实薪资", y="预测薪资", alpha=0.45)
min_value = min(plot_best["真实薪资"].min(), plot_best["预测薪资"].min())
max_value = max(plot_best["真实薪资"].max(), plot_best["预测薪资"].max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.title(f"{best_model_name}真实薪资与预测薪资对比")
plt.xlabel("真实薪资")
plt.ylabel("预测薪资")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "22_最优回归模型真实值预测值对比.png"), dpi=300)
plt.close()

# =========================
# 17. 最优模型残差散点图
# =========================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_best, x="预测薪资", y="预测误差", alpha=0.45)
plt.axhline(y=0, linestyle="--")
plt.title(f"{best_model_name}残差散点图")
plt.xlabel("预测薪资")
plt.ylabel("残差：真实值-预测值")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "23_最优回归模型残差散点图.png"), dpi=300)
plt.close()

# =========================
# 18. 最优模型残差分布图
# =========================
plt.figure(figsize=(10, 6))
sns.histplot(best_pred_df["预测误差"], bins=50, kde=True)
plt.title(f"{best_model_name}残差分布图")
plt.xlabel("残差：真实值-预测值")
plt.ylabel("频数")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "24_最优回归模型残差分布图.png"), dpi=300)
plt.close()

# =========================
# 19. 预测绝对误差箱线图
# =========================
plt.figure(figsize=(10, 5))
sns.boxplot(x=best_pred_df["绝对误差"])
plt.title(f"{best_model_name}预测绝对误差箱线图")
plt.xlabel("绝对误差")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "25_最优回归模型绝对误差箱线图.png"), dpi=300)
plt.close()

# =========================
# 20. 前200个样本真实值与预测值折线图
# =========================
line_df = best_pred_df.head(200).copy()
line_df["样本序号"] = range(1, len(line_df) + 1)

plt.figure(figsize=(14, 6))
plt.plot(line_df["样本序号"], line_df["真实薪资"], label="真实薪资", linewidth=1.8)
plt.plot(line_df["样本序号"], line_df["预测薪资"], label="预测薪资", linewidth=1.8)
plt.title(f"{best_model_name}前200个样本真实薪资与预测薪资对比")
plt.xlabel("样本序号")
plt.ylabel("薪资")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "26_最优回归模型前200样本预测对比.png"), dpi=300)
plt.close()

# =========================
# 21. 回归模型特征重要性
# =========================
importance_model_name = None

for candidate in ["XGBoost回归", "随机森林回归", "梯度提升回归", "决策树回归"]:
    if candidate in trained_pipelines:
        importance_model_name = candidate
        break

if importance_model_name is not None:
    importance_pipeline = trained_pipelines[importance_model_name]
    reg_model = importance_pipeline.named_steps["model"]

    try:
        feature_names = importance_pipeline.named_steps["preprocessor"].get_feature_names_out()
        feature_names = [name.replace("类别变量__", "").replace("数值变量__", "") for name in feature_names]

        if hasattr(reg_model, "feature_importances_"):
            importances = reg_model.feature_importances_

            importance_df = pd.DataFrame({
                "特征": feature_names,
                "重要性": importances
            }).sort_values(by="重要性", ascending=False).head(20)

            importance_df.to_excel(os.path.join(model_dir, "回归模型特征重要性.xlsx"), index=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x="重要性", y="特征")
            plt.title(f"{importance_model_name}特征重要性Top20")
            plt.xlabel("重要性")
            plt.ylabel("特征")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "27_回归模型特征重要性Top20.png"), dpi=300)
            plt.close()

    except Exception as e:
        print("特征重要性图生成失败：", e)

print("回归模型增强版运行完成！")
print("图表保存位置：", output_dir)
print("模型结果保存位置：", model_dir)
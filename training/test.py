import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# ===============================
# 1. 載入資料 & 處理 "--"
# ===============================
dataset = pd.read_csv('data/' + os.listdir('data')[2])
df = pd.DataFrame(dataset)

df.replace('--', np.nan, inplace=True)
df['DO'] = pd.to_numeric(df['DO'], errors='coerce')
df = df.dropna(subset=['DO']).reset_index(drop=True)

# ===============================
# 2. 指定欄位
# ===============================
cheap = ['Temperature','PH','EC','Degree','RPI']
categorical = ['Level','Code']
expensive = ['BOD','COD','NH3-N','TP','Turbidity','TN']
target = 'DO'

# ===============================
# 3. train/test split
# ===============================
df[expensive] = df[expensive].apply(pd.to_numeric, errors='coerce')
df[expensive] = df[expensive].fillna(df[expensive].median())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train_expensive = train_df[expensive].astype(float).copy()
X_test_expensive = test_df[expensive].astype(float).copy()

X_train_cheap = train_df[cheap + categorical].copy()
X_test_cheap = test_df[cheap + categorical].copy()
y_train = train_df[target].astype(float).values
y_test = test_df[target].astype(float).values

# ===============================
# 4. OneHot + 缺值填補
# ===============================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, cheap),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])

# ===============================
# 5. Baseline: Cheap-only model
# ===============================
cheap_only_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

cheap_only_model.fit(X_train_cheap, y_train)
y_pred_cheap = cheap_only_model.predict(X_test_cheap)
print("Cheap-only R2:", r2_score(y_test, y_pred_cheap),
      "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_cheap)))

# ===============================
# 6. Two-stage model
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
stage1_oof = np.zeros((len(X_train_cheap), len(expensive)))
stage1_test = np.zeros((len(X_test_cheap), len(expensive)))
stage1_models = {}

for i, col in enumerate(expensive):
    print(f"Training Stage1 for {col}...")
    stage1_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    oof = cross_val_predict(stage1_model, X_train_cheap, X_train_expensive[col].values, cv=kf, n_jobs=-1)
    stage1_oof[:, i] = oof
    stage1_model.fit(X_train_cheap, X_train_expensive[col].values)
    stage1_models[col] = stage1_model
    stage1_test[:, i] = stage1_model.predict(X_test_cheap)
    print(f"{col} OOF R2:", r2_score(X_train_expensive[col].values, oof))

# Stage2 input
stage2_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, cheap),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])

X_train_cheap_transformed = stage2_preprocessor.fit_transform(X_train_cheap)
X_test_cheap_transformed = stage2_preprocessor.transform(X_test_cheap)

X_train_stage2 = np.hstack([X_train_cheap_transformed, stage1_oof])
X_test_stage2 = np.hstack([X_test_cheap_transformed, stage1_test])

stage2_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
stage2_model.fit(X_train_stage2, y_train)
y_pred_stage2 = stage2_model.predict(X_test_stage2)

print("Two-stage Test R2:", r2_score(y_test, y_pred_stage2),
      "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_stage2)))

# --- 評估 Train/Test ---
y_train_pred_stage2 = stage2_model.predict(X_train_stage2)
print("\n🔹 Two-stage 訓練集")
print("R2:", r2_score(y_train, y_train_pred_stage2))
print("MAE:", mean_absolute_error(y_train, y_train_pred_stage2))
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_stage2)))

print("\n🔹 Two-stage 測試集")
print("R2:", r2_score(y_test, y_pred_stage2))
print("MAE:", mean_absolute_error(y_test, y_pred_stage2))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_stage2)))

# ===============================
# 7. Full-feature model (cheap + expensive)
# ===============================
X_train_full = pd.concat([X_train_cheap, X_train_expensive], axis=1)
X_test_full = pd.concat([X_test_cheap, X_test_expensive], axis=1)

full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, cheap + expensive),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])

full_model = Pipeline(steps=[
    ('preprocessor', full_preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

full_model.fit(X_train_full, y_train)
y_pred_full = full_model.predict(X_test_full)

print("\nFull-feature R2:", r2_score(y_test, y_pred_full),
      "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_full)))

# ===============================
# 8. 儲存模型
# ===============================
# joblib.dump(stage1_models, "stage1_models.joblib")
# joblib.dump(stage2_model, "two_stage_model.joblib")
# joblib.dump(cheap_only_model, "cheap_only_model.joblib")
# joblib.dump(full_model, "full_feature_model.joblib")

print("\n✅ 模型已儲存完成")

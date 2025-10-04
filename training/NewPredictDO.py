# 需要套件：scikit-learn, pandas, numpy, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# --- 使用者自訂區 ---
# 請把欄位名換成你的資料名稱
cheap = ['Temperature','EC','PH']
expensive = ['NH3_N','COD','BOD']  # 若某些 expensive 不存在就移掉
target = 'DO'

dataset = pd.read_csv('data/' + os.listdir('data')[2])
df = pd.DataFrame(dataset)
# --------------------

# 1) train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train_cheap = train_df[cheap].copy()
X_train_expensive = train_df[expensive].copy()
y_train = train_df[target].values

X_test_cheap = test_df[cheap].copy()
X_test_expensive = test_df[expensive].copy()  # 只用來評估；部署時你通常沒有
y_test = test_df[target].values

# --------------- 第一階段：對每個 expensive 用 cheap 訓練預測器 ---------------
# 我們要為訓練集產生 OOF 預測（避免用真實 expensive 訓練第二階段）
oof_preds = np.zeros((len(X_train_cheap), len(expensive)))
test_stage1_preds = np.zeros((len(X_test_cheap), len(expensive)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
stage1_models = {}

for i, col in enumerate(expensive):
    col_y = X_train_expensive[col].values
    # 使用 cross_val_predict 取得訓練集的 OOF predictions
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # OOF preds for training
    oof = cross_val_predict(model, X_train_cheap, col_y, cv=kf, n_jobs=-1, method='predict')
    oof_preds[:, i] = oof
    # 訓練在整個訓練集上（以便產生測試集預測）
    model.fit(X_train_cheap, col_y)
    stage1_models[col] = model
    # 對測試集做預測
    test_stage1_preds[:, i] = model.predict(X_test_cheap)
    # 評估該 expensive 的預測力
    r2_col = r2_score(X_train_expensive[col].values, oof)
    print(f"Stage1 predict {col}: OOF R2 = {r2_col:.3f}")

# --------------- 第二階段：用 cheap + stage1 OOF preds 訓練 DO 模型 ---------------
# 把 cheap 與 OOF preds 合併成第二階段的訓練特徵
X_train_stage2 = np.hstack([X_train_cheap.values, oof_preds])
X_test_stage2 = np.hstack([X_test_cheap.values, test_stage1_preds])

stage2_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
stage2_model.fit(X_train_stage2, y_train)

# 預測並評估
y_pred_stage2 = stage2_model.predict(X_test_stage2)
r2_stage2 = r2_score(y_test, y_pred_stage2)
rmse_stage2 = np.sqrt(mean_squared_error(y_test, y_pred_stage2))
print(f"Two-stage R2 (test): {r2_stage2:.3f}, RMSE: {rmse_stage2:.3f}")

# --------------- 比較 baseline：cheap-only model ---------------
cheap_only = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
cheap_only.fit(X_train_cheap, y_train)
y_pred_cheap = cheap_only.predict(X_test_cheap)
r2_cheap = r2_score(y_test, y_pred_cheap)
rmse_cheap = np.sqrt(mean_squared_error(y_test, y_pred_cheap))
print(f"Cheap-only R2 (test): {r2_cheap:.3f}, RMSE: {rmse_cheap:.3f}")

# ---------------（如果有 full-feature 可比較）full-feature 上界 ---------------
if set(expensive).issubset(df.columns):
    X_train_full = train_df[cheap + expensive].values
    X_test_full = test_df[cheap + expensive].values
    rf_full = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_full.fit(X_train_full, y_train)
    y_pred_full = rf_full.predict(X_test_full)
    print("Full-feature R2 (test):", r2_score(y_test, y_pred_full))

# --------------- 儲存模型（部署用） ---------------
joblib.dump(stage1_models, 'stage1_models.joblib')  # dict of models
joblib.dump(stage2_model, 'stage2_model.joblib')
joblib.dump(cheap_only, 'cheap_only_model.joblib')   # baseline, optional

# --------------- 範例：部屬時的預測函數 ---------------
def predict_two_stage(new_df):
    """
    new_df: dataframe with only cheap columns
    returns: predicted_DO (numpy)
    """
    # load if you didn't keep models in memory
    # stage1_models = joblib.load('stage1_models.joblib')
    # stage2_model = joblib.load('stage2_model.joblib')

    # 1) 用 stage1_models 逐一預測 expensive
    pred_exp = np.zeros((len(new_df), len(stage1_models)))
    for i, col in enumerate(stage1_models.keys()):
        pred_exp[:, i] = stage1_models[col].predict(new_df[cheap])
    # 2) 合併 cheap + pred_exp 並用 stage2_model 預測 DO
    X_stage2 = np.hstack([new_df[cheap].values, pred_exp])
    y_do = stage2_model.predict(X_stage2)
    return y_do

# 使用示例:
# new_samples_df = pd.DataFrame({...})
# preds = predict_two_stage(new_samples_df)

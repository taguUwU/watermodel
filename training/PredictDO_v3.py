import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ========== 1. 資料讀取與預處理 ==========

# 載入檔案（第3個檔案）
dataset = pd.read_csv('data/' + os.listdir('data')[2])
df = pd.DataFrame(dataset)

# 填補缺失值（只針對有用欄位）
cols = ['Temperature', 'EC', 'PH', 'Salinity', 'DO']
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols)

# 建立衍生特徵
data['Temp^2'] = data['Temperature'] ** 2
data['EC/PH'] = data['EC'] / (data['PH'] + 1e-5)
data['EC*Temp'] = data['EC'] * data['Temperature']

# 特徵與目標分離
X = data.drop(columns=['DO'])
y = data['DO']

# ========== 2. 切分資料集 ==========

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 3. 隨機森林 + 參數搜尋 + KFold ==========

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [150],       # 提升模型穩定性
    'max_depth': [20],            # 放寬深度，學到更多複雜關係
    'min_samples_split': [5],      # 減少分裂門檻，讓樹更活躍
    'min_samples_leaf': [2]        # 葉節點樣本變少，能捕捉細節
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print("最佳參數：", grid_search.best_params_)

# ========== 4. 預測與評估 ==========

def evaluate_model(model, X, y, dataset_name="資料集"):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    print(f"\n=== {dataset_name} 表現 ===")
    print("R²:", round(r2, 4))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    return r2, mae, rmse

evaluate_model(best_rf, X_train, y_train, "訓練集")
evaluate_model(best_rf, X_test, y_test, "測試集")

# ========== 5. 特徵重要性列印 ==========

feature_importances = pd.Series(
    best_rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n📊 特徵重要性（由高到低）：")
print(feature_importances)

# ========== 6. 模型儲存 ==========

joblib.dump(best_rf, 'PredictDO.pkl')
print("\n✅ 模型已儲存為 PredictDO.pkl")

# ========== 顯示相關係數 ==========

print("\n📈 各特徵與 DO 的相關係數（Pearson r）:")
corr_matrix = data.corr()
print(corr_matrix['DO'].sort_values(ascending=False))
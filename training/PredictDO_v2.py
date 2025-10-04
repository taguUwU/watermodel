import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 讀取資料（假設只有一個檔案）
dataset = pd.read_csv('data/' + os.listdir('data')[2])
df = pd.DataFrame(dataset)

# 填補缺失值
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(df[['Temperature', 'EC', 'PH', 'DO']])
data = pd.DataFrame(data, columns=['Temperature', 'EC', 'PH', 'DO'])

# 特徵與目標
X = data.drop(columns=['DO'])
y = data['DO']

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型與網格搜尋
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100],         # 稍多一點穩定性
    'max_depth': [8, 10],          # 深度適中，不會過擬合
    'min_samples_split': [4, 6],   # 增加每次分裂需要的樣本量
    'min_samples_leaf': [3, 4]     # 葉節點不能太小
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最佳模型
best_rf = grid_search.best_estimator_
print("最佳參數：", grid_search.best_params_)

# 預測（訓練集與測試集）
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# 訓練集指標
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5

# 測試集指標
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5

# 顯示結果
print("\n=== 訓練集表現 ===")
print("R²:", round(r2_train, 4))
print("MAE:", round(mae_train, 4))
print("RMSE:", round(rmse_train, 4))

print("\n=== 測試集表現 ===")
print("R²:", round(r2_test, 4))
print("MAE:", round(mae_test, 4))
print("RMSE:", round(rmse_test, 4))

# 儲存模型
joblib.dump(best_rf, 'PredictDO.pkl')
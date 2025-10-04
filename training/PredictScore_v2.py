import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 讀取資料（假設只有一個檔案）
dataset = pd.read_csv('data/' + os.listdir('data')[0])
df = pd.DataFrame(dataset)

# 填補缺失值（針對你使用的特徵與目標欄位）
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(df[['Temperature', 'EC', 'PH', 'DO', 'Score']])
data = pd.DataFrame(data, columns=['Temperature', 'EC', 'PH', 'DO', 'Score'])

# 特徵與目標
X = data.drop(columns=['Score'])
y = data['Score']

# 資料切分（80% 訓練、20% 測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林模型
rf = RandomForestRegressor(random_state=42)

# 網格搜尋參數（可擴充）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最佳模型
best_rf = grid_search.best_estimator_
print("最佳參數：", grid_search.best_params_)

# 預測
y_pred = best_rf.predict(X_test)

# 效能指標
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("隨機森林預測表現")
print("R²:", round(r2, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# 儲存模型
joblib.dump(best_rf, 'random_forest_ccme.pkl')

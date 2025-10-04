import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer

# 讀取資料
dataset = pd.read_csv('data/' + os.listdir('data')[0])
df = pd.DataFrame(dataset)

# 補值
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(df[['Temperature', 'EC', 'PH', 'DO', 'Score']])
data = pd.DataFrame(np.round(data, 2), columns=['Temperature', 'EC', 'PH', 'DO', 'Score'])

# 特徵與目標
X = data.drop(columns=['Score'])
y = data['Score'].values.ravel()

print(f"特徵形狀: {X.shape}, 樣本數: {X.shape[0]}")

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建立管線
model = Pipeline([
    ("poly_features", PolynomialFeatures()),
    ("std_scaler", StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])

# 參數搜尋空間
param_grid = {
    'poly_features__degree': [2, 3, 4],
    'lasso__alpha': [0.001, 0.01, 0.1, 1, 10]
}

# GridSearchCV 訓練
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 輸出結果
print("最佳參數：", grid_search.best_params_)
print("最佳交叉驗證得分：", grid_search.best_score_)

# 可視化不同參數組合的分數
results = pd.DataFrame(grid_search.cv_results_)
pivot = results.pivot(index='param_poly_features__degree', columns='param_lasso__alpha', values='mean_test_score')
print("交叉驗證分數表：\n", pivot)

# 使用最佳參數訓練最終模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 效能指標
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("測試集 R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
'''
# 可視化：預測 vs 實際
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真實 Score')
plt.ylabel('預測 Score')
plt.title('預測值 vs 真實值')
plt.grid(True)
plt.tight_layout()
plt.show()

# 保存模型
joblib.dump(best_model, 'best_model.pkl')
'''

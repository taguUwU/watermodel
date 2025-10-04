import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# === 載入資料 ===
df = pd.read_csv('data/' + os.listdir('data')[2])  # 輸入檔，需含 Temperature, pH, Salinity, EC

# === 特徵與目標 ===
X = df[['Temperature', 'EC', 'PH', 'Salinity']].values

# 若有真實 DO，就用它訓練
y = df['DO'].values if 'DO' in df.columns else None

# === 特徵標準化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 建立 ANN 模型 ===
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))  # 論文用單隱藏層、10 個神經元
model.add(Dense(1))  # 輸出 DO
model.compile(optimizer='adam', loss='mean_squared_error')

# === 模型訓練 ===
if y is not None:
    model.fit(X_scaled, y, epochs=200, verbose=0)

# === 預測 DO ===
df['DO_ANN'] = model.predict(X_scaled).flatten()

# === 儲存結果與模型 ===
#df.to_csv('predicted_DO.csv', index=False)
model.save('ann_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("完成：已預測 DO,儲存至 predicted_DO.csv")
print("模型儲存為 ann_model.h5,scaler 儲存為 scaler.pkl")

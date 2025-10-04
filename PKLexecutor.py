from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import serial
import time
import requests

ser = serial.Serial('COM4', 115200, timeout=1)
time.sleep(2)

# 載入已訓練的模型
ScoreModel   = joblib.load('NewPredictScore0.975.pkl')
stage1Models = joblib.load('stage1_model.joblib')
stage2Model  = joblib.load('stage2_model.joblib')
stage2Pre    = joblib.load('stage2_preprocessor.joblib')
keepCols     = joblib.load('stage1_keep_cols.joblib')

buffer = []  # 用來存儲四行數據

def align_categorical(cheap_df, preprocessor):
    cheap_df_aligned = cheap_df.copy()
    ohe_categories = preprocessor.named_transformers_['cat'].categories_
    cat_cols = preprocessor.transformers_[1][2]

    for col, cats in zip(cat_cols, ohe_categories):
        val = cheap_df_aligned[col].iloc[0]
        if val not in cats:
            cheap_df_aligned[col] = cats[0]  # 使用訓練資料第一個類別
    return cheap_df_aligned

while True:
    try:
        # 讀取 Arduino 的輸出 (放大鏡)
        data = ser.readline().decode().strip()
        
        if data:
            print("收到數據",data)
            buffer.append(data)
            
            if len(buffer) == 4:
                try:
                    # 讀取的資料
                    # 25 °C
                    # 2000 μs/cm
                    # 1.8 V
                    
                    temperature = float(buffer[0].split(" ")[0])
                    ec =          float(buffer[1].split(" ")[0])
                    voltage =     float(buffer[2].split(" ")[0])
                    ph =          float(buffer[3].split(" ")[0])
                    
                    # ===========================================
                    
                    # 預測Do的資料
                    cheap_df = pd.DataFrame([{
                        "Temperature": temperature,
                        "PH": ph,
                        "EC": ec,
                        "Degree": 0.0,      # 沒有感測器，用 0 或 median
                        "RPI": 0.0,         # 同上
                        "Level": "66",      # 用一個 dummy 類別
                        "Code": "1006"        # 同上
                    }])

                    # 對齊 categorical
                    cheap_df_aligned = align_categorical(cheap_df, stage2Pre)

                    # Stage1 預測保留特徵
                    stage1_preds = []
                    for col in keepCols:
                        stage1_preds.append(stage1Models[col].predict(cheap_df_aligned)[0])
                    stage1_preds = np.array(stage1_preds, dtype=float).reshape(1, -1)

                    # Stage2 輸入
                    X_cheap_trans = stage2Pre.transform(cheap_df_aligned)
                    if hasattr(X_cheap_trans, "toarray"):
                        X_cheap_trans = X_cheap_trans.toarray()
                    if X_cheap_trans.ndim == 1:
                        X_cheap_trans = X_cheap_trans.reshape(1, -1)

                    X_stage2 = np.hstack([X_cheap_trans, stage1_preds])

                    # 預測 DO
                    do = float(stage2Model.predict(X_stage2)[0])
                    do = round(do, 2)

                    # ===========================================
                    
                    # 預測分數的資料
                    ScoreData = pd.DataFrame({
                        "Temperature": [temperature],
                        "EC": [ec],
                        "PH": [ph],
                        "DO": [do]
                    })
                    
                    # 使用RF模型進行預測
                    score = f"{ScoreModel.predict(ScoreData)[0]:.2f}"
                        
                    # 目前時間
                    now_time = datetime.now().strftime('%Y-%m-%d %H:%M')
                    
                    # 顯示預測結果
                    print("預測數據", do, "mg/L")
                    print("預測結果", score, "Score")
                    print("目前時間", now_time, '\n')
                    payload = {
                        "temperature": temperature,
                        "ec": ec,
                        "ph": ph,
                        "do": do,
                        "score": score,
                        "now_time": now_time
                    }
                    
                    # 用 POST 送到 FastAPI
                    response = requests.post(
                        "http://127.0.0.1:8001/receive-data",  # 後端 POST /receive-data
                        json=payload
                    )
                    time.sleep(1)
                    
                except Exception as e:
                    print("解析錯誤:", e)
                
                buffer = []
                
    except Exception as e:
        print("發生錯誤:", e)
        break
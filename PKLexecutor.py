from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import serial #arduino -> python
import time
import requests
import json

ser = serial.Serial('COM4', 115200, timeout=1)
time.sleep(2)

# 載入已訓練的模型
ScoreModel = joblib.load('NewPredictScore0.975.pkl')
DoModel = joblib.load('PredictDO0.6055.pkl')
buffer = []  # 用來存儲四行數據
history_scores = [] # 存儲歷史分數

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
                    # 9.23 V
                    # 
                    temperature = float(buffer[0].split(" ")[0])
                    ec = float(buffer[1].split(" ")[0])
                    voltage = float(buffer[2].split(" ")[0])
                    ph = float(buffer[3].split(" ")[0])
                    
                    # 預測Do的資料
                    DoData = pd.DataFrame({
                        "Temperature": [temperature],
                        "EC": [ec],
                        "PH": [ph],
                        "Salinity": [temperature*0.00064]
                    })
                    DoData['Temp^2'] = DoData['Temperature'] ** 2
                    DoData['EC/PH'] = DoData['EC'] / (DoData['PH'] + 1e-5)
                    DoData['EC*Temp'] = DoData['EC'] * DoData['Temperature']
                    do = f"{DoModel.predict(DoData)[0]:.2f}"
                    
                    # 預測分數的資料
                    ScoreData = pd.DataFrame({
                        "Temperature": [temperature],
                        "EC": [ec],
                        "PH": [ph],
                        "DO": [float(do)]
                    })
                    
                    # 使用RF模型進行預測
                    score = f"{ScoreModel.predict(ScoreData)[0]:.2f}"
                    
                    # 保存歷史分數
                    history_scores.append(float(score))
                    if len(history_scores) > 10:  # 保留最新50筆資料
                        history_scores.pop(0)
                        
                    # 預測未來趨勢
                    predicted_score = [0]
                    if len(history_scores) >= 5:
                        x = np.arange(len(history_scores))
                        y = np.array(history_scores)

                        coeffs = np.polyfit(x, y, 1)
                        
                        future_x = np.arange(len(history_scores), len(history_scores)+20)
                        predicted_score = np.around(np.polyval(coeffs, future_x), 2).tolist()
                    
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
                    time.sleep(2)
                    
                except:
                    print("解析錯誤")
                
                buffer = []
                
    except:
        print("發生錯誤")
        break
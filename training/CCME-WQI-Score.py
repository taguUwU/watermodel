'''
CCME-WQI calculator
DO PH TP(Temperature) EC Salinity Turbidity

Standard:
TP 12~25C
EC 500~1400µS/cm
PH 7~8
DO 6.5~8 mg/L
Salinity < 0.5 ppt
Turbidity 0~60 NTU
'''

import math as m
import pandas as pd

# TP EC PH DO
df = pd.read_excel("FinalData.xlsx", sheet_name=0)

def calculate(row):
    TP = row['Temperature']
    EC = row['EC']
    PH = row['PH']
    DO = row['DO']

    # 判定合格/不合格
    ExcursionDO = 0
    ExcursionPH = 0
    ExcursionTP = 0
    ExcursionEC = 0
    correct=0
    error=0
    if DO < 6.5:
        error += 1
        ExcursionDO = 1 - (DO / 6.5)
    elif DO > 8:
        error += 1
        ExcursionDO = (DO - 8) - 1 
    else:
        correct += 1

    if PH < 7:
        error += 1
        ExcursionPH = 1 - (PH / 7)
    elif PH > 8:
        error += 1
        ExcursionPH = (PH - 8) - 1
    else:
        correct += 1

    if TP < 12:
        error += 1
        ExcursionTP = 1 - (TP / 12)
    elif TP > 25:
        error += 1
        ExcursionTP = (TP - 25) - 1
    else:
        correct += 1
        
    if EC < 500:
        error += 1
        ExcursionEC = 1 - (EC / 500)
    elif EC > 1400:
        error += 1
        ExcursionEC = (EC - 1400) - 1
    else:
        correct += 1

    # F1（Scope）不合格參數比例
    f1 = error / (correct + error) * 100

    # F2（Frequency）不合格次數 / 總檢測次數
    f2 = error / (correct + error) * 100

    # F3（Amplitude）超標程度
    ExcursionALL = ExcursionDO + ExcursionPH + ExcursionTP + ExcursionEC
    nse =  ExcursionALL / 4
    f3 = nse / (0.01 * nse + 0.01)

    # 計算 CCME WQI
    score = 100 - m.sqrt((f1**2 + f2**2 + f3**2) / m.sqrt(3))
    return round(score, 2)

df['Score'] = df.apply(calculate, axis=1)
df.to_excel("output.xlsx", index=False)
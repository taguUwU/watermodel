import joblib
import pandas as pd
DoData = pd.DataFrame({
    "Temperature": [25],
    "EC": [1000],
    "PH": [7],
    "Salinity": [1000*0.00064]
})

DoData['Temp^2'] = DoData['Temperature'] ** 2
DoData['EC/PH'] = DoData['EC'] / (DoData['PH'] + 1e-5)
DoData['EC*Temp'] = DoData['EC'] * DoData['Temperature']

DoModel = joblib.load('PredictDO0.6055.pkl')
print(DoModel.predict(DoData))
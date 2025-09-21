from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# 允許跨域請求（CORS設定）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 內存資料存儲
history_data = []

# 定義資料模型
class SensorData(BaseModel):
    temperature: float
    ec: float
    ph: float
    do: float
    score: float
    now_time: str

# 接收資料端點
@app.post("/receive-data")
async def create_sensor_data(data: SensorData):
    try:
        record = data.dict()
        print(data)
        history_data.append(record)
        return {"status": "success", "data": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 獲取歷史資料端點（與前端對接）
@app.get("/history")
async def get_history():
    try:
        return history_data[-10:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",              
        host="0.0.0.0",
        port=8001,
        reload=True
    )

'''
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
received_data = []  # 暫存收到的資料

# 允許 Expo Go 的請求來源（開發時通常用本地 IP）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應替換為具體域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義收到的資料格式
class DataItem(BaseModel):
    temperature: float
    ec: float
    ph: float
    do: float
    score: float

@app.post("/receive-data")
async def receive_data(item: DataItem):
    #print("收到資料：", item)
    received_data.clear()
    received_data.append(item.dict())
    return {
        "data": item.dict()
    }

@app.get("/history")
async def get_history():
    return received_data

if __name__ == "__main__":
    uvicorn.run("main:app",              
        host="0.0.0.0",
        port=8001,
        reload=True)
    
#                                                      __----~~~~~~~~~~~------___
#                                     .  .   ~~//====......          __--~ ~~
#                     -.            \_|//     |||\\  ~~~~~~::::... /~
#                  ___-==_       _-~o~  \/    |||  \\            _/~~-
#          __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
#      _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
#    .~       .~       |   \\ -_    /  /-   /   ||      \   /
#   /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
#   |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
#            '         ~-|      /|    |-~\~~       __--~~
#                        |-~~-_/ |    |   ~\_   _-~            /\
#                             /  \     \__   \/~                \__
#                         _--~ _/ | .-~~____--~-/                  ~~==.
#                        ((->/~   '.|||' -_|    ~~-/ ,              . _||
#                                   -_     ~\      ~~---l__i__i__i--~~_/
#                                   _-~-__   ~)  \--______________--~~
#                                 //.-~~~-~_--~- |-------~~~~~~~~
#                                        //.-~~~--\
#                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 神獸保佑，程式碼沒Bug!
'''
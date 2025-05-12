from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

# Load model
with open("expsmooth_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="API Prediksi Pemakaian Solar",
    description="Prediksi jumlah pemakaian solar menggunakan model Exponential Smoothing berdasarkan input jumlah hari ke depan.",
    version="1.0.0"
)

class InputData(BaseModel):
    steps: int  # jumlah hari ke depan

@app.post("/predict")
def predict(data: InputData):
    forecast = model.forecast(data.steps)
    
    # Ambil tanggal sekarang dengan zona waktu Samarinda (WITA / Asia/Makassar)
    current_date = datetime.now(ZoneInfo("Asia/Makassar"))
    
    prediction_list = []
    for i, value in enumerate(forecast):
        forecast_date = current_date + timedelta(days=i)
        formatted_date = forecast_date.strftime("%Y-%m-%d")
        prediction_list.append(f"Prediksi pada {formatted_date} : {round(value, 2)} KL")
    
    total = round(sum(forecast), 2)
    total_text = f"Total prediksi selama {data.steps} hari adalah: {total} KL"
    
    prediction_list.append(total_text)

    return {
        "prediction": prediction_list
    }

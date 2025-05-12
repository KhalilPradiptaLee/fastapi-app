from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fastapi.middleware.cors import CORSMiddleware
import os
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        response = {"message": "Hello from Python on Vercel!"}
        self.wfile.write(json.dumps(response).encode())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "expsmooth_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="API Prediksi Pemakaian Solar",
    description="Prediksi jumlah pemakaian solar menggunakan model Exponential Smoothing berdasarkan input jumlah hari ke depan.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti "*" dengan domain frontend kalau sudah produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

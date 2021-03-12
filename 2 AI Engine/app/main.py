import os
os.system("python -m pip install python-multipart")

import json
from typing import List
from fastapi import Depends, FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil

app = FastAPI()

#To override CROS origion problem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd() + '/myfirstproject-305412-bd26f6fbb24b.json'

#default api to show that continer is up
@app.get("/")
async def root():
    return {"message": "Speech Recognition System is started and listening for commands!"}



#Install required libraries
@app.get("/install_dependencies/")
def install_dependencies_api():
	try:
		os.system("apt-get update")
		os.system("apt-get upgrade")
		os.system("apt-get -y install apt-utils gcc libpq-dev libsndfile-dev")
		os.system("python -m pip install --upgrade pip")
		os.system("python -m pip install sklearn")
		os.system("python -m pip install numpy")
		os.system("python -m pip install numba")
		os.system("python -m pip install librosa")
		os.system("python -m pip install keras")
		os.system("python -m pip install --no-cache-dir --default-timeout=100 --upgrade tensorflow")
		os.system("python -m pip install --upgrade google-cloud-storage")
		os.system("python -m pip install --upgrade google-cloud-bigquery")
		return {"message": "Installed!"}
	except Exception as e:
		return {"message" : str(e)}



#Import other modules and methods
@app.get("/init/")
def init_api():
	try:
		from .model import initialize_model, predict_audio
		from .download_dataset import download_data
		return {"message": "Initialization complete!"}
	except Exception as e:
		return {"message" : str(e)}



#Greet user
@app.get("/hello/")
async def hello_api(name: str):
	return {"message":f"hello {name} from model container!"}



#Download dataset
@app.get("/download/")
def download_api(root: str, table: str):
	try:
		res = download_data(root, table)
		return {"message" : res}
	except Exception as e:
		return {"message" : str(e)}



#Initilize model
@app.get("/init_model/")
def init_model_api(already_trained: bool):
	try:
		summary = initialize_model(already_trained)
		return {"message" : summary}
	except Exception as e:
		return {"message" : str(e)}



#Upload file to predict
@app.post("/upload_file/")
def upload_file_api(audio: UploadFile = File(...)):
	try:
		with open("app/predict.wav", "wb") as buffer:
			shutil.copyfileobj(audio.file, buffer)
		return {"message": audio.filename}
	except Exception as e:
		return {"message" : str(e)}		


#Predict the uploaded file
@app.get("/predict/")
def predict_api():
	try:
		audio, sample_rate = librosa.load('app/predict.wav', sr = 16000)
		prediction = predict_audio(audio, sample_rate)
		return {"message": prediction}
	except Exception as e:
		return {"message" : str(e)}
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from src.scripts.clean_hc3_data import clean_hc3_data
from src.scripts.load_hc3 import load_hc3
from src.scripts.split_hc3_data import split_hc3_data


app = FastAPI(title="HumanOrAI API", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))



@app.get("/predict")
def predict(text: str = "") -> dict[str, str]:
	if not text.strip():
		return {"message": "Please enter some text first."}

	return {"message": "Text received.", "text": text}



@app.get("/pipeline")
def run_pipeline():

	load_hc3()
	clean_hc3_data()
	split_hc3_data()

	return {"message": "HC3 pipeline completed successfully."}
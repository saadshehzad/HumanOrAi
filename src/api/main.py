from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="HumanOrAI API", version="0.1.0")



@app.get("/predict")
def predict() -> dict[str, str]:
	return {"message": "Hello, World!"}
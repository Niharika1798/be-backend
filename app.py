from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from tensorflow import keras as K
import pandas as pd
import json

model = K.models.load_model('model')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.debug = True

class Features(BaseModel):
    features: list

@app.post("/detect")
async def detect_chances_of_osteo(features: Features):
    print(features.features)
    x = pd.DataFrame(features.features).T
    res = model.predict(x)
    return json.dumps(res.tolist()[0][0])


if __name__ == '__main__': 
    uvicorn.run(app)
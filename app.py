from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from tensorflow import keras as K
import pandas as pd
import json

model = K.models.load_model('model')

app = FastAPI()

class OsteoDetails(BaseModel):
    features: list

@app.post("/detect")
async def detect_chances_of_osteo(details: OsteoDetails):
    x = pd.DataFrame(details.features).T
    res = model.predict(x)
    return json.dumps(res)


if __name__ == '__main__': 
    uvicorn.run(app)
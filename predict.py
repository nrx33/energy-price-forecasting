from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd

# Load the model from the uploaded script (assuming it is serialized using pickle)
with open('best_model.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# Create an instance of FastAPI
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    demand: float
    demand_pos_RRP: float
    demand_neg_RRP: float
    min_temperature: float
    max_temperature: float
    solar_exposure: float
    rainfall: float
    frac_at_neg_RRP: float
    month: int
    school_day: int

# Define the endpoint for prediction
@app.post("/predict")
async def predict(input_data: InputData):
    # Prepare the data for prediction
    data = pd.DataFrame([{
        'demand': input_data.demand,
        'demand_pos_RRP': input_data.demand_pos_RRP,
        'demand_neg_RRP': input_data.demand_neg_RRP,
        'min_temperature': input_data.min_temperature,
        'max_temperature': input_data.max_temperature,
        'solar_exposure': input_data.solar_exposure,
        'rainfall': input_data.rainfall,
        'frac_at_neg_RRP': input_data.frac_at_neg_RRP,
        'month': input_data.month,
        'school_day': input_data.school_day
    }])

    # Make prediction
    prediction = model.predict(data)

    # Return the prediction result
    return {"Predicted RRP": f"${prediction.tolist()[0]:.2f}"}

# Run the application if this script is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

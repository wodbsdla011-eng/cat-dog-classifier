from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from .model import predict_image

app = FastAPI(title="Cat vs Dog Image Classifier", description="A simple MLOps demo pipeline.")

@app.get("/")
def read_root():
    return {"message": "Server is running! Navigate to /docs for the Swagger UI to test the API."}

@app.post("/predict")
async def predict_cat_dog(file: UploadFile = File(...)):
    # Simple validation using content-type
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not an image."})
    
    # Read the bytes of the file for prediction
    contents = await file.read()
    prediction_result = predict_image(contents)
    
    if prediction_result.startswith("Error"):
        return JSONResponse(status_code=500, content={"error": prediction_result})

    return {
        "filename": file.filename,
        "prediction": prediction_result
    }

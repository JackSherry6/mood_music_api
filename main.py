from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import librosa
import numpy as np
import joblib
import tempfile
import os

app = FastAPI()

model = joblib.load("mood_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

from extract_features import extract_features

emotion_names = ["Angry","Calm","Cheerful","Dark","Dreamy","Exciting","Groovy","Melancholy","Romantic","Sad","Sexy","Soft","Epic","Warm","Cold"]

@app.post("/predict")
async def predict_mood(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Your file is not an MP3")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    features_df = extract_features(tmp_path)
    os.unlink(tmp_path)

    if features_df is None:
        raise HTTPException(status_code=400, detail="Audio is unreadable")

    features_flat = pd.DataFrame()
    for col in features_df.columns:
        if isinstance(features_df[col].iloc[0], list):
            expanded = pd.DataFrame(features_df[col].iloc[0]).T
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            features_flat = pd.concat([features_flat, expanded], axis=1)
        else:
            features_flat[col] = features_df[col]

    X_scaled = scaler.transform(features_flat)
    predictions = model.predict(X_scaled)[0] 

    mood_scores = {name: float(score) for name, score in zip(emotion_names, predictions)}
    return mood_scores

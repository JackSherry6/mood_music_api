# Mood Music API
This project is a FastAPI-based web application that leverages machine learning to analyze MP3 audio files and predict the emotional content of songs. Using librosa, a Python library for audio analysis, the application extracts a wide range of audio features such as spectral, rhythmic, and harmonic characteristics from the uploaded track.

The extracted features are processed and scaled before being fed into a custom-trained multi-output regression model, which generates a quantitative emotional profile for each song across 15 different distinct moods. The model is trained to handle multiple outputs simultaneously, allowing it to capture complex emotional nuances in music.

# Features
- FastAPI backend: lightweight and fast API server for handling audio uploads and predictions.
- Audio feature extraction: leverages librosa to capture rich features from songs.
- Multi-output regression model: predicts multiple emotional dimensions simultaneously.
- JSON output: structured, machine-readable predictions for integration into other systems.

# Usage
1. Clone this repo
2. Install dependencies:
   - Python 3.11
   - uvicorn 0.38.0
   - aiofiles 25.1.0
   - python-multipart 0.0.20
   - librosa 0.11.0
   - numpy 2.3.4
   - pandas 2.3.3
   - joblib 1.5.2
   - fastapi 0.121.0

3. Run the API locally with uvicorn

   - ```uvicorn main:app --reload```

4. Open this link in your browser:

   - http://127.0.0.1:8000/docs

5. Run program using /predict and upload your mp3 file

# Limitations
While this project demonstrates the ability to predict emotional content from audio features, the model’s accuracy is inherently limited by the size and diversity of the training data. Music emotion recognition is a complex problem that typically requires large-scale datasets with thousands of labeled tracks to capture the wide variety of genres, styles, and listener perceptions; with a smaller dataset, predictions should be interpreted as illustrative rather than fully reliable. If Spotify improved their API this would be achievable, but as of now they are not responding to any of my inquiries about my api key failing.

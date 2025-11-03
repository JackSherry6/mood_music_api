## **Issue with ai model file being too large, currently being resolved
# music_mood_api
A FastAPI-based machine learning project that analyzes mp3 files and predicts the emotional content of songs based on audio features extracted using librosa. Built with a custom-trained multi-output regression model.

# Features
- Upload .mp3 files and get emotion predictions (Anger, Sadness, Excitement, etc..)
- Emotion predictions are based on a multi-output regression model
- Fast, simple REST API built with FastAPI
- Fully testable and ready for deployment (e.g., Render, Hugging Face)

# Setting up
1. Clone the repo:
   - git clone https://github.com/JackSherry6/music_mood_api.git
   - cd emotion-music-api

2. Install dependencies:
   
   Make sure you're using python 3.9 or above and install:
     - pip install -r requirements.txt

3. Run the API locally

   - uvicorn main:app --reload

4. Visit in your browser:

   - http://127.0.0.1:8000/docs

5. Run program using /predict and upload your mp3 file

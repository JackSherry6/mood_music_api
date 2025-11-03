# test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    with open("test_song.mp3", "rb") as f:
        response = client.post("/predict", files={"file": ("test_song.mp3", f, "audio/mpeg")})
    
    assert response.status_code == 200
    assert "predicted_emotions" in response.json()

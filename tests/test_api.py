from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "potato"}

def test_predict():
    response = client.post("/predict", json={"input_text": "Sample product description"})
    assert response.status_code == 200
    assert "prediction" in response.json()

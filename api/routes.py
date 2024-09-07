from fastapi import APIRouter
from .models import InferenceRequest, InferenceResponse
import torch
from transformers import BertTokenizer
from scripts.transformer import TransformerModel

router = APIRouter()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TransformerModel()
model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval()

@router.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    inputs = tokenizer(request.input_text, return_tensors="pt", padding='max_length', max_length=128, truncation=True)
    
    with torch.no_grad():
        output = model(**inputs)

    probs = torch.nn.functional.softmax(output, dim=-1)
    predicted_class = probs.argmax(dim=-1).item()

    prediction = str(predicted_class)

    return InferenceResponse(prediction=prediction)

client = MongoClient("mongodb://localhost:27017/")
db = client['your_database']
collection = db['your_collection']

@router.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    prediction = "predicted_value"
    collection.insert_one({"input": request.input_data, "prediction": prediction})
    return InferenceResponse(prediction=prediction)

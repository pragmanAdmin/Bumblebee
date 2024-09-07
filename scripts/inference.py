import torch
from transformers import BertTokenizer
from pymongo import MongoClient
from scripts.transformer import Transformer

client = MongoClient('potato')
db = client.bumblebee

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_vocab_size=30522,
    trg_vocab_size=30522,
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=512,
    num_layers=3,
    forward_expansion=4,
    heads=8,
    dropout=0.1,
    device=device,
    max_len=100
).to(device)

model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

input_data = db.attribute_test_data.find_one({}, {'_id': False})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert input data to model input format
input_description = input_data['title']
input_tokens = tokenizer.encode(input_description, max_length=100, truncation=True, padding='max_length')

input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

print("Prediction:", output)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from pymongo import MongoClient

from scripts.transformer import Transformer
from scripts.dataset import CustomDataset
from scripts.utils import train_model

client = MongoClient('potato')
db = client.bumblebee

train_data = pd.DataFrame(list(db.attribute_train_data.find({}, {'_id': False})))
train_solution = pd.DataFrame(list(db.attribute_train_solution.find({}, {'_id': False})))
val_data = pd.DataFrame(list(db.attribute_val_data.find({}, {'_id': False})))
val_solution = pd.DataFrame(list(db.attribute_val_solution.find({}, {'_id': False})))

train_merged = pd.merge(train_data, train_solution, on='indoml_id')
val_merged = pd.merge(val_data, val_solution, on='indoml_id')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Special tokens and maximum length for tokenization
max_len = 100
pad_idx = tokenizer.pad_token_id
bos_idx = tokenizer.cls_token_id
eos_idx = tokenizer.sep_token_id

def tokenize_text(text):
    tokens = tokenizer.encode(text, max_length=max_len, truncation=True, padding='max_length')
    return tokens

# Tokenization of the datasets
train_merged['src_tokens'] = train_merged['title'].apply(tokenize_text)
train_merged['trg_tokens'] = train_merged['L0_category'].apply(lambda x: tokenize_text(str(x)))

val_merged['src_tokens'] = val_merged['title'].apply(tokenize_text)
val_merged['trg_tokens'] = val_merged['L0_category'].apply(lambda x: tokenize_text(str(x)))

# Dataset and DataLoader Creation
train_dataset = CustomDataset(train_merged['src_tokens'].tolist(), train_merged['trg_tokens'].tolist())
val_dataset = CustomDataset(val_merged['src_tokens'].tolist(), val_merged['trg_tokens'].tolist())

batch_size = 32
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
src_vocab_size = tokenizer.vocab_size
trg_vocab_size = tokenizer.vocab_size
embed_size = 512
num_layers = 3
heads = 8
dropout = 0.1
forward_expansion = 4
learning_rate = 3e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, and loss function
model = Transformer(
    src_vocab_size=src_vocab_size, 
    trg_vocab_size=trg_vocab_size, 
    src_pad_idx=pad_idx, 
    trg_pad_idx=pad_idx, 
    embed_size=embed_size, 
    num_layers=num_layers, 
    forward_expansion=forward_expansion, 
    heads=heads, 
    dropout=dropout, 
    device=device, 
    max_len=max_len
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training
train_model(model, train_iterator, val_iterator, optimizer, criterion, num_epochs, device)

# Save the model
torch.save(model.state_dict(), 'model.pth')
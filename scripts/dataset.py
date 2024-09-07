import pymongo
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, db_name, collection_name, tokenizer):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.tokenizer = tokenizer
        self.data = list(self.collection.find())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['description']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

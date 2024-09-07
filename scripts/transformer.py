import torch
import torch.nn as nn
import math
from transformers import BertModel

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos_encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        return x + self.pos_encoding[:x.size(0), :]

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value), attn_weights

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % heads == 0
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        N = query.size(0)
        query_len, key_len, value_len = query.size(1), key.size(1), value.size(1)

        query = self.query(query).view(N, query_len, self.heads, self.head_dim)
        key = self.key(key).view(N, key_len, self.heads, self.head_dim)
        value = self.value(value).view(N, value_len, self.heads, self.head_dim)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        attention, _ = scaled_dot_product_attention(query, key, value, mask)
        attention = attention.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)

        return self.fc_out(attention)

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion=4):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, expansion * embed_size)
        self.fc2 = nn.Linear(expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(self.position_embedding(self.word_embedding(x)))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# Decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_len)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.dropout(self.position_embedding(self.word_embedding(x)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

# Full Transformer Model incorporating BERT
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6,
                 forward_expansion=4, heads=8, dropout=0.1, device="cuda", max_len=100):
        super(TransformerModel, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_len)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # Using BERT to get embeddings
        bert_embeddings = self.bert(src)[0]  # BERT sequence output
        enc_src = self.encoder(bert_embeddings, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
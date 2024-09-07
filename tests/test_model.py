import torch
from scripts.transformer import TransformerModel

def test_model_loading():
    model = TransformerModel()
    model.load_state_dict(torch.load('model.pth'))
    assert model is not None

# def test_inference():
#     model = TransformerModel()
#     model.eval()
#     dummy_input = torch.randint(0, 1000, (1, 128))
#     with torch.no_grad():
#         output = model(dummy_input)
#     assert output is not None

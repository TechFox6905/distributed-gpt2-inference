import pytest
import torch
from model.gpt import GPT, GPTConfig

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32
    )

@pytest.fixture
def model(small_config):

    device = "cpu"

    model = GPT(small_config)
    model.to(device)
    model.eval()
    
    return model
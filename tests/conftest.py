import pytest
import torch
from gpt.core.model import GPT2, GPT2Config

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.fixture
def small_config():
    return GPT2Config(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32
    )

@pytest.fixture
def model(small_config):

    device = "cpu"

    model = GPT2(small_config)
    model.to(device)
    model.eval()
    
    return model
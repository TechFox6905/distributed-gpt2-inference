import torch

from app.generate import generate

def test_generation_length(model):

    idx = torch.randint(0, 100, (1, 5))

    out = generate(
        model,
        idx,
        max_tokens=10
    )

    assert out.shape[1] == 15


def test_generation_stops_on_eos(model):

    eos = 99

    idx = torch.tensor([[1, 2, eos]])

    out = generate(
        model,
        idx,
        max_tokens=5,
        eos_token_id=eos
    )

    assert out[0, -1] == eos
import torch

from gpt.core.generate import sample_logits

def test_sample_logits_shape():

    logits = torch.randn(2, 100)

    out = sample_logits(logits)

    assert out.shape == (2, 1)
    

def test_top_k_keeps_only_top_tokens():

    logits = torch.tensor([
        [1.0, 2.0, 3.0, 100.0]
    ])

    out = sample_logits(
        logits,
        top_k=1
    )

    assert out.item() == 3
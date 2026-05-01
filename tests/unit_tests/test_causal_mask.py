import torch

def test_future_tokens_do_not_affect_past(model):

    model.eval()

    x1 = torch.tensor([[1, 2, 3, 4]])
    x2 = torch.tensor([[1, 2, 3, 99]])

    logits1, _, _ = model(x1)
    logits2, _, _ = model(x2)

    assert torch.allclose(
        logits1[:, :3],
        logits2[:, :3],
        atol=1e-5
    )
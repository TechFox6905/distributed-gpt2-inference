import torch

def test_dropout_disabled_in_eval(model):

    model.eval()

    x = torch.randint(0, model.config.vocab_size, (1, 10))

    out1, _, _ = model(x)
    out2, _, _ = model(x)

    assert torch.allclose(out1, out2)
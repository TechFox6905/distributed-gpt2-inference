import torch

def test_forward_output_shape(model):
    x = torch.randint(0, model.config.vocab_size, (2, 16))

    logits, loss, cache = model(x)

    assert logits.shape == (2, 16, model.config.vocab_size)
    assert len(cache) == model.config.n_layer
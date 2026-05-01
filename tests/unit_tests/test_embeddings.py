import torch

def test_token_embedding_shape(model):

    idx = torch.randint(0, model.config.vocab_size, (4, 10))

    tok_emb = model.transformer.wte(idx)

    assert tok_emb.shape == (
        4,
        10,
        model.config.n_embd
    )


def test_position_embedding_shape(model):

    pos = torch.arange(0, 10)

    pos_emb = model.transformer.wpe(pos)

    assert pos_emb.shape == (
        10,
        model.config.n_embd
    )
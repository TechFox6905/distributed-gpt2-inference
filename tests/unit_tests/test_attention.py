import torch

def test_attention_output_shape(model):

    x = torch.randn(2, 8, model.config.n_embd)

    block = model.transformer.h[0]

    out, present = block.attn(x)

    # Output shape preserved
    assert out.shape == x.shape

    # KV cache structure
    k, v = present

    assert k.shape == (
        2,
        model.config.n_head,
        8,
        model.config.n_embd // model.config.n_head
    )

    assert v.shape == (
        2,
        model.config.n_head,
        8,
        model.config.n_embd // model.config.n_head
    )
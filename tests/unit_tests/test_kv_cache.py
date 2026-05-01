import torch

def test_kv_cache_matches_full_forward(model):

    model.eval()

    idx = torch.randint(0, 100, (1, 10))

    # Full forward
    full_logits, _, _ = model(idx)

    # Incremental forward
    cache = None

    for i in range(idx.size(1)):

        token = idx[:, i:i+1]

        logits, _, cache = model(
            token,
            past_kvs=cache
        )

    incremental_logits = logits

    assert torch.allclose(
        full_logits[:, -1],
        incremental_logits[:, -1],
        atol=1e-5
    )
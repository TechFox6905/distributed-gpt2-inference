import time
import torch

from app.generate import generate

def test_generation_speed(model):

    model.eval()

    idx = torch.randint(0, 100, (1, 32))

    start = time.time()

    generate(
        model,
        idx,
        max_tokens=32
    )

    elapsed = time.time() - start

    assert elapsed < 10
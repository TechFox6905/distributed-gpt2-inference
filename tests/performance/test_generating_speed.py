import time
import torch

from gpt.core.generate import generate

def test_generation_speed(model):

    idx = torch.randint(0, 100, (1, 8))

    start = time.time()

    generate(
        model,
        idx,
        max_tokens=8
    )

    elapsed = time.time() - start

    assert elapsed < 10
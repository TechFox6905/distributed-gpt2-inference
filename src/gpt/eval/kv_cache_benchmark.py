import time
import torch
import json

from gpt.core.model import GPT2
from pathlib import Path
from transformers import GPT2Tokenizer

@torch.no_grad()
def benchmark_without_cache(
    model,
    input_ids,
    max_new_tokens,
    device
):

    model.eval()

    x = input_ids.to(device)

    # warmup
    _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(max_new_tokens):

        logits, _, _ = model(x)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        x = torch.cat([x, next_token], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


@torch.no_grad()
def benchmark_with_cache(
    model,
    input_ids,
    max_new_tokens,
    device
):

    model.eval()

    x = input_ids.to(device)

    # warmup
    _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    logits, _, past_kvs = model(x)

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )

    for _ in range(max_new_tokens - 1):

        logits, _, past_kvs = model(
            next_token,
            past_kvs=past_kvs
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


def save_results(results, path=r"src\gpt\results\kv_cache_benchmarks.json"):

    path = Path(path)

    # create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2.from_pretrained("gpt2")

    model.to(device)

    prompt = "Artificial intelligence is"

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    )

    max_new_tokens = 100

    without_cache = benchmark_without_cache(
        model,
        input_ids,
        max_new_tokens,
        device
    )

    with_cache = benchmark_with_cache(
        model,
        input_ids,
        max_new_tokens,
        device
    )

    speedup = without_cache / with_cache

    results = {
        "without_cache_sec": round(without_cache, 4),
        "with_cache_sec": round(with_cache, 4),
        "speedup": round(speedup, 2)
    }

    print(results)

    save_results(results)

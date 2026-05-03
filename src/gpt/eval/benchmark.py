import json
import time
import torch

from gpt.core.generate import generate
from gpt.core.model import GPT2
from pathlib import Path
from transformers import GPT2Tokenizer


@torch.no_grad()
def benchmark_generation(
    model,
    input_ids,
    max_new_tokens=100,
    num_runs = 5,
    device="cuda"
):

    model.eval()

    input_ids = input_ids.to(device)

    # warmup
    _ = generate(
        model=model,
        tokens=input_ids,
        max_tokens=10
    )

    latencies = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(num_runs):

        start = time.perf_counter()

        # generation
        output = generate(
            model=model,
            tokens=input_ids,
            max_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

        latencies.append(end - start)

    avg_latency = sum(latencies) / len(latencies)
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = generated_tokens / avg_latency

    return {
        "latency_sec": round(avg_latency, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "device": device,
        "prompt_tokens": input_ids.shape[1],
        "generated_tokens": generated_tokens
    }


def save_results(results, path=r"src\gpt\results\generation_benchmark.json"):
    path = Path(path)

    # create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2.from_pretrained("gpt2")

    model.to(device)
    model.eval()

    prompt = "The future of artificial intelligence"

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    )

    results = benchmark_generation(
        model,
        input_ids,
        max_new_tokens=100,
        device=device
    )

    print(results)

    save_results(results)
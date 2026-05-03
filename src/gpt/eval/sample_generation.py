import torch

from gpt.core.model import GPT2
from transformers import GPT2Tokenizer
from gpt.core.generate import generate

@torch.no_grad()
def generate_sample(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    device="cuda"
):

    model.eval()

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    output = generate(
        model=model,
        tokens=input_ids,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return text, temperature, top_k, top_p


def save_generation(
    prompt,
    output,
    temperature,
    top_k,
    top_p,
    path=r"src\gpt\results\generations.md"
):

    content = f"""
# Sample Generation

## Prompt

{prompt}

## Parameters

```text
temperature = {temperature}
top_k = {top_k}
top_p = {top_p}
```

## Output

{output}

---

"""

    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2.from_pretrained("gpt2")

    model.to(device)

    prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "Deep learning models are"
    ]

    # clear previous generations
    with open(r"src\gpt\results\generations.md", "w"):
        pass

    for prompt in prompts:

        output, temperature, top_k, top_p = generate_sample(
            model,
            tokenizer,
            prompt,
            device=device
        )

        print("\nPROMPT:", prompt)
        print("\nOUTPUT:", output)

        save_generation(prompt, output, temperature, top_k, top_p)
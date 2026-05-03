import math
import json
import torch

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset

from gpt.core.model import GPT2


@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in tqdm(dataloader, desc="Evaluating"):

        x = x.to(device)
        y = y.to(device)

        _, loss, _ = model(x, y)

        batch_tokens = y.numel()

        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = float("inf") if avg_loss > 20 else math.exp(avg_loss)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4)
    }


def save_results(results, path=r"src\gpt\results\perplexity.json"):
    path = Path(path)

    # create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def build_wikitext2_dataloader(
    tokenizer,
    block_size=128,
    batch_size=4
):

    # ---------------------------------
    # Load WikiText-2 validation split
    # ---------------------------------

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation"
    )

    # ---------------------------------
    # Merge text
    # ---------------------------------

    text = "\n\n".join(dataset["text"])

    # ---------------------------------
    # Tokenize
    # ---------------------------------

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    # ---------------------------------
    # Build causal LM samples
    # ---------------------------------

    x_list = []
    y_list = []

    for i in range(0, len(tokens) - block_size - 1, block_size):

        x = tokens[i:i + block_size]
        y = tokens[i + 1:i + block_size + 1]

        x_list.append(x)
        y_list.append(y)

    x_tensor = torch.tensor(x_list)
    y_tensor = torch.tensor(y_list)

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load tokenizer
    # -----------------------------

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # -----------------------------
    # Load model
    # -----------------------------

    model = GPT2.from_pretrained("gpt2")

    model.to(device)
    model.eval()

    # ---------------------------------
    # Build WikiText-2 dataloader
    # ---------------------------------

    dataloader = build_wikitext2_dataloader(
        tokenizer=tokenizer,
        block_size=128,
        batch_size=4
    )

    # -----------------------------
    # Run evaluation
    # -----------------------------

    results = evaluate(model, dataloader, device)

    print("\nEvaluation Results")
    print(results)

    save_results(results)
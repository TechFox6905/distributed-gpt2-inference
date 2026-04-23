import torch
import torch.nn.functional as F

def sample_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Apply temperature, top-k, and top-p sampling to logits.
    """

    # 🔹 Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # 🔹 Top-K
    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    # 🔹 Top-P (Nucleus Sampling)
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # mask tokens beyond top_p
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits[sorted_mask] = -float("inf")

        # restore original order
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    # 🔹 Final sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(
    model,
    tokens,
    max_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    past = None

    for _ in range(max_tokens):

        if past is None:
            logits, _, past = model(tokens)
        else:
            logits, _, past = model(tokens[:, -1:], past_kvs=past)

        logits = logits[:, -1, :]

        next_token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens
import torch

def generate(model, tokens, max_tokens):

    for _ in range(max_tokens):

        logits, _ = model(tokens)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens
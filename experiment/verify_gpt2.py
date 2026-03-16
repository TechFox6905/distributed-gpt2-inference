import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from gpt import GPT  # import your model

device = "cuda" if torch.cuda.is_available() else "cpu"

# prompt
prompt = "The future of artificial intelligence is"

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

# load your GPT model
model = GPT.from_pretrained("gpt2").to(device)
model.eval()

# load huggingface GPT
hf_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
hf_model.eval()

with torch.no_grad():

    # your model
    logits_custom, _ = model(tokens)

    # huggingface model
    hf_out = hf_model(tokens)
    logits_hf = hf_out.logits

# compare logits
diff = torch.abs(logits_custom - logits_hf).mean()

print("Average Logit Difference:", diff.item())

# check next token prediction
custom_next = torch.argmax(logits_custom[:, -1, :])
hf_next = torch.argmax(logits_hf[:, -1, :])

print("Custom next token:", custom_next.item())
print("HF next token:", hf_next.item())

print("Custom token:", tokenizer.decode([custom_next]))
print("HF token:", tokenizer.decode([hf_next]))

# generate text with your model
idx = tokens

for _ in range(40):

    logits, _ = model(idx)
    logits = logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    idx = torch.cat((idx, next_token), dim=1)

print(tokenizer.decode(idx[0]))
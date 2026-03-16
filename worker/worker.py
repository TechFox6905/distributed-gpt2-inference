import redis
import json
import torch
from transformers import GPT2Tokenizer
from app.model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT.from_pretrained("gpt2")
model = model.to(device)
model.eval()

r = redis.Redis(host="redis", port=6379, decode_responses=True)

while True:

    task = r.brpop("llm_queue")
    data = json.loads(task[1])

    task_id = data["id"]
    prompt = data["prompt"]

    print(f"Processing {task_id}")

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(data["max_tokens"]):

        with torch.no_grad():
            logits, _ = model(tokens)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        tokens = torch.cat((tokens, next_token), dim=1)

    generated_text = tokenizer.decode(tokens[0])

    r.set(f"result:{task_id}", generated_text)

    print(f"Finished {task_id}")
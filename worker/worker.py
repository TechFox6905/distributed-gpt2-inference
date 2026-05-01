import os
import json
import time
import redis
import torch
from transformers import GPT2Tokenizer
from app.model import GPT2
from app.generate import generate

# -------- Device --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Model --------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2.from_pretrained("gpt2")
model = model.to(device)
model.eval()
print("Model loaded:", model is not None)

# -------- Redis --------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_timeout=5,
)

print("Worker ready. Waiting for tasks...")


# -------- Safe Redis --------
def safe_redis(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except redis.RedisError:
            print("Redis error, retrying in 2s...")
            time.sleep(2)

# --------- Batching --------
BATCH_SIZE = 8
BATCH_TIMEOUT = 0.01  # 10ms

def get_batch(r):
    batch = []

    # blocking wait for first task
    task = r.brpop("llm_queue", 0)
    if task:
        batch.append(json.loads(task[1]))

    start = time.time()

    # collect more tasks (non-blocking)
    while len(batch) < BATCH_SIZE and (time.time() - start) < BATCH_TIMEOUT:
        task = r.lpop("llm_queue")
        if task:
            batch.append(json.loads(task))
        else:
            break

    return batch

def process_batch(batch):

    task_ids = []
    prompts = []
    max_tokens_list = []
    temps, topks, topps = [], [], []

    for data in batch:
        task_ids.append(data["id"])
        prompts.append(data["prompt"])
        max_tokens_list.append(min(int(data.get("max_tokens", 50)), 200))
        temps.append(data.get("temperature", 1.0))
        topks.append(data.get("top_k"))
        topps.append(data.get("top_p"))

        safe_redis(r.setex, f"status:{data['id']}", 300, "processing")

    # 🔹 Tokenize (pad to same length)
    encoded = [tokenizer.encode(p)[:1024] for p in prompts]
    max_len = max(len(x) for x in encoded)

    padded = [
        x + [tokenizer.eos_token_id] * (max_len - len(x))
        for x in encoded
    ]

    tokens = torch.tensor(padded, dtype=torch.long).to(device)  # (B, T)

    # 🔹 For simplicity → use max of max_tokens
    max_tokens = max(max_tokens_list)

    with torch.no_grad():
        out_tokens = generate(
            model,
            tokens,
            max_tokens=max_tokens,
            temperature=temps[0],   # simplified (see note below)
            top_k=topks[0],
            top_p=topps[0]
        )

    # 🔹 Decode individually
    for i, task_id in enumerate(task_ids):
        text = tokenizer.decode(out_tokens[i], skip_special_tokens=True)

        result = {"id": task_id, "output": text}

        safe_redis(r.setex, f"result:{task_id}", 300, json.dumps(result))
        safe_redis(r.setex, f"status:{task_id}", 300, "done")

        print(f"[DONE] {task_id}")


# -------- Worker Loop --------
def worker_loop():

    while True:

        batch = get_batch(r)

        if not batch:
            continue

        print(f"[BATCH] size={len(batch)}")

        process_batch(batch)


if __name__ == "__main__":
    worker_loop()
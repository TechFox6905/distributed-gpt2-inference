import os
import json
import time
import redis
import torch
from transformers import GPT2Tokenizer
from gpt.core.model import GPT2
from gpt.core.generate import generate

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/app/cache")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

def load_model():

    global tokenizer, model

    print("=" * 60)
    print("Starting model initialization")
    print(f"Device: {device}")
    print(f"HF cache: {HF_CACHE_DIR}")
    print("=" * 60)

    start = time.time()

    print("[1/5] Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        token=HF_TOKEN,
        cache_dir=HF_CACHE_DIR
    )

    print("[2/5] Loading GPT weights...")
    model = GPT2.from_pretrained(
        "gpt2",
        token=HF_TOKEN,
        cache_dir=HF_CACHE_DIR
    )

    print("[3/5] Moving model to device...")
    model = model.to(device)

    print("[4/5] Setting eval mode...")
    model.eval()

    elapsed = time.time() - start

    print("[5/5] Model ready")
    print(f"Startup time: {elapsed:.2f}s")
    print("=" * 60)

# -------- Redis --------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
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
    load_model()
    worker_loop()
    print("Worker ready. Waiting for tasks...")
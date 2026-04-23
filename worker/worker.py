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


# -------- Worker Loop --------
while True:

    try:
        task = r.brpop("llm_queue", 0)  # block forever
    except redis.RedisError as e:
        print(f"Redis connection lost: {e}")
        time.sleep(2)
        continue

    if task is None:
        continue
    
    try:
        data = json.loads(task[1])
        task_id = data["id"]
        prompt = data["prompt"]
        max_tokens = min(int(data.get("max_tokens", 50)), 200)

        print(f"[START] {task_id}")

        # Update status
        safe_redis(r.setex, f"status:{task_id}", 300, "processing")

        # Tokenize (limit input size)
        tokens = tokenizer.encode(prompt, return_tensors="pt")[:, :1024]
        tokens = tokens.to(device)

        # Inference
        with torch.no_grad():
            tokens = generate(model, tokens, max_tokens=max_tokens)

        generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

        result = {
            "id": task_id,
            "output": generated_text
        }

        # Store result + status
        safe_redis(r.setex, f"result:{task_id}", 300, json.dumps(result))
        safe_redis(r.setex, f"status:{task_id}", 300, "done")

        print(f"[DONE] {task_id}")

    except Exception as e:
        print(f"[ERROR] {e}")

        if 'task_id' in locals():
            safe_redis(
                r.setex,
                f"result:{task_id}",
                300,
                json.dumps({"error": str(e)})
            )
            safe_redis(r.setex, f"status:{task_id}", 300, "failed")
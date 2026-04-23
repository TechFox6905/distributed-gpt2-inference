import os
import json
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis

app = FastAPI()

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5
)

# -------- Models --------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_tokens: int = Field(default=50, ge=1, le=200)
    temperature: float = Field(default=1.0, ge=0.1, le=5.0)
    top_k: int = Field(default=None, ge=1, le=100)
    top_p: float = Field(default=None, ge=0.0, le=1.0)


# -------- Helpers --------
def safe_redis_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except redis.RedisError:
        raise HTTPException(status_code=500, detail="Redis unavailable")


# -------- API --------
@app.post("/generate")
def generate(req: GenerateRequest):

    task_id = str(uuid.uuid4())

    payload = {
        "id": task_id,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p
    }

    # Push to queue
    safe_redis_call(r.rpush, "llm_queue", json.dumps(payload))

    # Set initial status
    safe_redis_call(r.setex, f"status:{task_id}", 300, "queued")

    return {"task_id": task_id}


@app.get("/result/{task_id}")
def get_result(task_id: str):

    # Check result
    result = safe_redis_call(r.get, f"result:{task_id}")

    if result:
        safe_redis_call(r.delete, f"result:{task_id}")
        safe_redis_call(r.delete, f"status:{task_id}")
        return json.loads(result)

    # Check status
    status = safe_redis_call(r.get, f"status:{task_id}")

    if status:
        return {"status": status}

    return {"status": "not_found"}

@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "ok"}
    except redis.RedisError:
        return {"status": "redis_down"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
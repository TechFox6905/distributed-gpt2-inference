from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json
import uuid

app = FastAPI()

r = redis.Redis(host="redis", port=6379, decode_responses=True)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.post("/generate")
def generate(req: GenerateRequest):

    task_id = str(uuid.uuid4())

    payload = {
        "id": task_id,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens
    }

    r.lpush("llm_queue", json.dumps(payload))

    return {
        "task_id": task_id,
        "status": "queued"
    }


@app.get("/result/{task_id}")
def get_result(task_id: str):

    result = r.get(f"result:{task_id}")

    if result is None:
        return {
            "task_id": task_id,
            "status": "processing"
        }

    return {
        "task_id": task_id,
        "status": "completed",
        "result": result
    }
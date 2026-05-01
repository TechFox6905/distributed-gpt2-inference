import json
import fakeredis
import threading
import time

from app.worker import get_batch



def test_get_batch_respects_batch_size(monkeypatch):

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    # Push 20 tasks
    for i in range(20):

        payload = {
            "id": str(i),
            "prompt": f"hello {i}"
        }

        fake_redis.rpush(
            "llm_queue",
            json.dumps(payload)
        )

    # monkeypatch redis instance if needed
    batch = get_batch(fake_redis)

    assert len(batch) <= 8



def test_get_batch_returns_tasks():

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    payload = {
        "id": "1",
        "prompt": "hello"
    }

    fake_redis.rpush(
        "llm_queue",
        json.dumps(payload)
    )

    batch = get_batch(fake_redis)

    assert len(batch) == 1

    assert batch[0]["prompt"] == "hello"



def test_empty_queue_timeout():

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    results = []

    def run():
        batch = get_batch(fake_redis)
        results.append(batch)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

    time.sleep(0.1)

    # should still be waiting
    assert len(results) == 0
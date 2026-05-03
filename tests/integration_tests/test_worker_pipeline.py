import json
import fakeredis
import torch

from unittest.mock import MagicMock

from gpt.worker.worker import process_batch

def test_process_batch_stores_results(monkeypatch):

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    # mock redis
    monkeypatch.setattr(
        "gpt.worker.worker.r",
        fake_redis
    )

    # mock tokenizer
    mock_tokenizer = MagicMock()

    mock_tokenizer.encode.return_value = [1, 2, 3]

    mock_tokenizer.decode.return_value = "generated text"

    monkeypatch.setattr(
        "gpt.worker.worker.tokenizer",
        mock_tokenizer
    )

    # mock generate()
    def fake_generate(*args, **kwargs):

        return torch.tensor([
            [1, 2, 3, 4]
        ])

    monkeypatch.setattr(
        "gpt.worker.worker.generate",
        fake_generate
    )

    batch = [
        {
            "id": "abc",
            "prompt": "hello",
            "max_tokens": 5,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None
        }
    ]

    process_batch(batch)

    result = fake_redis.get("result:abc")

    assert result is not None

    parsed = json.loads(result)

    assert parsed["id"] == "abc"

    assert parsed["output"] == "generated text"



def test_process_batch_updates_status(monkeypatch):

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr(
        "gpt.worker.worker.r",
        fake_redis
    )

    monkeypatch.setattr(
        "gpt.worker.worker.tokenizer",
        MagicMock(
            encode=lambda x: [1,2],
            decode=lambda x, skip_special_tokens=True: "ok"
        )
    )

    monkeypatch.setattr(
        "gpt.worker.worker.generate",
        lambda *args, **kwargs: torch.tensor([[1,2,3]])
    )

    batch = [{
        "id": "task1",
        "prompt": "hi",
        "max_tokens": 5
    }]

    process_batch(batch)

    status = fake_redis.get("status:task1")

    assert status == "done"
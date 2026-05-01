import fakeredis

def test_queue_push():

    r = fakeredis.FakeRedis(decode_responses=True)

    r.rpush("llm_queue", "task")

    assert r.llen("llm_queue") == 1
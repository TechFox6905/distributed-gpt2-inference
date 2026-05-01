from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health():

    response = client.get("/health")

    assert response.status_code == 200


def test_generate_endpoint():

    response = client.post(
        "/generate",
        json={
            "prompt": "hello"
        }
    )

    assert response.status_code == 200

    assert "task_id" in response.json()
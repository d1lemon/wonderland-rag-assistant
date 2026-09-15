import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "Wonderland RAG Assistant API"
    assert data["version"] == "0.1.0"

    assert response.headers["X-Request-ID"].startswith("req_")
    assert "X-Process-Time-Ms" in response.headers


def test_ready_returns_503_when_rag_service_not_loaded():
    response = client.get("/ready")

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == "RAG service is still loading."


def test_chat_rejects_question_that_is_too_short():
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "Hi",
            "top_k": 2
        }
    )

    assert response.status_code == 422


def test_chat_rejects_top_k_above_limit():
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "Who does Alice follow down the rabbit-hole?",
            "top_k": 10
        }
    )

    assert response.status_code == 422


def test_chat_rejects_missing_question():
    response = client.post(
        "/api/v1/chat",
        json={
            "top_k": 2
        }
    )

    assert response.status_code == 422


def test_feedback_endpoint_accepts_valid_feedback():
    response = client.post(
        "/api/v1/feedback",
        json={
            "request_id": "req_example123",
            "rating": "up",
            "comment": "The chapter citation was helpful."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "recorded"
    assert data["request_id"] == "req_example123"
    assert data["rating"] == "up"
    assert data["feedback_id"].startswith("feedback_")

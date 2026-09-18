import sys
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.auth import get_current_user_id
from app.main import app


TEST_USER_ID = "11111111-1111-1111-1111-111111111111"
TEST_SESSION_ID = "22222222-2222-2222-2222-222222222222"
TEST_USER_MESSAGE_ID = "33333333-3333-3333-3333-333333333333"
TEST_ASSISTANT_MESSAGE_ID = "44444444-4444-4444-4444-444444444444"
TEST_FEEDBACK_ID = "55555555-5555-5555-5555-555555555555"


@pytest.fixture(autouse=True)
def isolate_fastapi_app():
    original_startup_handlers = list(app.router.on_startup)

    app.router.on_startup.clear()
    app.dependency_overrides.clear()

    yield

    app.dependency_overrides.clear()
    app.router.on_startup[:] = original_startup_handlers

def fake_current_user_id():
    return TEST_USER_ID


def make_rag_result(
    answer_status: str = "grounded",
    top_distance: float | None = 0.35,
    generation_latency_ms: int = 25,
) -> dict:
    return {
        "answer": (
            "Alice follows the White Rabbit. [Chapter I]\n\n"
            "Educational information only—not personalized advice."
        ),
        "retrieved_chunks": [
            {
                "chunk_id": "alice-chapter-I-chunk-000",
                "text": (
                    "Alice saw the White Rabbit run down "
                    "the rabbit-hole."
                ),
                "metadata": {
                    "chapter_number": "I",
                    "chapter_title": "Down the Rabbit-Hole",
"source_url": (
                        "https://www.gutenberg.org/"
                        "files/11/11-h/11-h.htm"
                    ),
                },
                "distance": top_distance or 0.0,
            }
        ],
        "latency_ms": 30,
        "retrieval_latency_ms": 5,
        "generation_latency_ms": generation_latency_ms,
        "top_retrieval_distance": top_distance,
        "answer_status": answer_status,
    }


def fake_session() -> dict:
    return {
        "id": TEST_SESSION_ID,
        "user_id": TEST_USER_ID,
        "title": "Who does Alice follow down the rabbit-hole?",
    }


def fake_user_message() -> dict:
    return {
        "id": TEST_USER_MESSAGE_ID,
        "session_id": TEST_SESSION_ID,
        "user_id": TEST_USER_ID,
        "role": "user",
        "content": "Who does Alice follow down the rabbit-hole?",
    }


def fake_assistant_message() -> dict:
    return {
        "id": TEST_ASSISTANT_MESSAGE_ID,
        "session_id": TEST_SESSION_ID,
        "user_id": TEST_USER_ID,
        "role": "assistant",
        "content": "Alice follows the White Rabbit.",
    }


def fake_feedback() -> dict:
    return {
        "id": TEST_FEEDBACK_ID,
        "message_id": TEST_ASSISTANT_MESSAGE_ID,
        "user_id": TEST_USER_ID,
        "rating": "up",
        "comment": "Helpful answer.",
        "created_at": "2026-09-18T00:00:00+00:00",
    }


def configure_test_app(monkeypatch, rag_result=None):
    main_module = sys.modules["app.main"]

    app.dependency_overrides[get_current_user_id] = (
        fake_current_user_id
    )

    monkeypatch.setattr(
        main_module,
        "rag_service",
        Mock(
            answer_question=Mock(
                return_value=rag_result or make_rag_result()
            )
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_chat_session",
        Mock(return_value=fake_session()),
    )

    monkeypatch.setattr(
        main_module,
        "get_user_session",
        Mock(return_value=fake_session()),
    )

    monkeypatch.setattr(
        main_module,
        "save_chat_message",
        Mock(
            side_effect=[
                fake_user_message(),
                fake_assistant_message(),
            ]
        ),
    )

    monkeypatch.setattr(
        main_module,
        "save_feedback",
        Mock(return_value=fake_feedback()),
    )


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_returns_503_when_rag_service_not_loaded(monkeypatch):
    main_module = sys.modules["app.main"]

    monkeypatch.setattr(main_module, "rag_service", None)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service is still loading."


def test_chat_requires_bearer_token(monkeypatch):
    configure_test_app(monkeypatch)

    app.dependency_overrides.clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": (
                    "Who does Alice follow down the rabbit-hole?"
                ),
                "top_k": 2,
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "A Bearer access token is required."
    )


def test_chat_rejects_question_that_is_too_short(monkeypatch):
    configure_test_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "Hi",
                "top_k": 2,
            },
        )

    assert response.status_code == 422


def test_chat_rejects_top_k_above_limit(monkeypatch):
    configure_test_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": (
                    "Who does Alice follow down the rabbit-hole?"
                ),
                "top_k": 5,
            },
        )

    assert response.status_code == 422


def test_chat_returns_authenticated_grounded_response(monkeypatch):
    configure_test_app(
        monkeypatch,
        rag_result=make_rag_result(
            answer_status="grounded",
            top_distance=0.35,
            generation_latency_ms=25,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": (
                    "Who does Alice follow down the rabbit-hole?"
                ),
                "top_k": 2,
            },
        )

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["answer_status"] == "grounded"
    assert body["session_id"] == TEST_SESSION_ID
    assert body["user_message_id"] == TEST_USER_MESSAGE_ID
    assert body["assistant_message_id"] == TEST_ASSISTANT_MESSAGE_ID
    assert body["top_retrieval_distance"] == 0.35
    assert body["generation_latency_ms"] == 25
    assert body["retrieved_chunk_count"] == 1


def test_chat_returns_guardrail_response(monkeypatch):
    guarded_answer = (
        "I do not have sufficient support in the provided source material "
        "to answer that.\n\n"
        "Educational information only—not personalized advice."
    )

    configure_test_app(
        monkeypatch,
        rag_result={
            "answer": guarded_answer,
            "retrieved_chunks": [],
            "latency_ms": 5,
            "retrieval_latency_ms": 0,
            "generation_latency_ms": 0,
            "top_retrieval_distance": None,
            "answer_status": "insufficient_evidence",
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is the capital of Japan?",
                "top_k": 2,
            },
        )

    body = response.json()

    assert response.status_code == 200
    assert body["answer_status"] == "insufficient_evidence"
    assert body["generation_latency_ms"] == 0
    assert body["top_retrieval_distance"] is None
    assert body["retrieved_chunk_count"] == 0


def test_feedback_requires_bearer_token(monkeypatch):
    configure_test_app(monkeypatch)

    app.dependency_overrides.clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            json={
                "message_id": TEST_ASSISTANT_MESSAGE_ID,
                "rating": "up",
                "comment": "Helpful answer.",
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "A Bearer access token is required."
    )


def test_feedback_endpoint_accepts_valid_feedback(monkeypatch):
    configure_test_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            json={
                "message_id": TEST_ASSISTANT_MESSAGE_ID,
                "rating": "up",
                "comment": "Helpful answer.",
            },
        )

    body = response.json()

    assert response.status_code == 200
    assert body["id"] == TEST_FEEDBACK_ID
    assert body["message_id"] == TEST_ASSISTANT_MESSAGE_ID
    assert body["rating"] == "up"


def test_feedback_rejects_invalid_rating(monkeypatch):
    configure_test_app(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            json={
                "message_id": TEST_ASSISTANT_MESSAGE_ID,
                "rating": "maybe",
            },
        )

    assert response.status_code == 422

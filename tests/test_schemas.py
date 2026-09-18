import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChatRequest,
    FeedbackRequest,
)


def test_chat_request_accepts_valid_input():
    request = ChatRequest(
        question="Who does Alice follow down the rabbit-hole?",
        top_k=2,
    )

    assert request.question.startswith("Who does Alice")
    assert request.top_k == 2
    assert request.session_id is None


def test_chat_request_rejects_short_question():
    with pytest.raises(ValidationError):
        ChatRequest(
            question="Hi",
            top_k=2,
        )


def test_chat_request_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        ChatRequest(
            question="Who does Alice follow down the rabbit-hole?",
            top_k=5,
        )


def test_feedback_request_accepts_allowed_rating():
    feedback = FeedbackRequest(
        message_id="00000000-0000-0000-0000-000000000000",
        rating="up",
        comment="Helpful answer.",
    )

    assert feedback.rating == "up"


def test_feedback_request_rejects_invalid_rating():
    with pytest.raises(ValidationError):
        FeedbackRequest(
            message_id="00000000-0000-0000-0000-000000000000",
            rating="maybe",
        )


def test_feedback_request_rejects_long_comment():
    with pytest.raises(ValidationError):
        FeedbackRequest(
            message_id="00000000-0000-0000-0000-000000000000",
            rating="down",
            comment="x" * 1001,
        )

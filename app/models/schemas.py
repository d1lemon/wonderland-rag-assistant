from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A question about Alice's Adventures in Wonderland.",
    )
    top_k: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of relevant source chunks to retrieve.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing user-owned session UUID. "
            "A new session is created when omitted."
        ),
    )


class SourceCitation(BaseModel):
    chunk_id: str
    chapter_number: str
    chapter_title: str
    source_url: str
    excerpt: str
    retrieval_distance: float


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunk_count: int
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    top_retrieval_distance: float | None
    answer_status: Literal["grounded", "insufficient_evidence"]
    status: str


class FeedbackRequest(BaseModel):
    message_id: str = Field(
        ...,
        description="UUID of an assistant message in the user's chat history.",
    )
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
    )


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    user_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None
    created_at: str

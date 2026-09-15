from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="A question about Alice's Adventures in Wonderland."
    )
    top_k: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of relevant source chunks to retrieve."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional future chat-session identifier."
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
    answer: str
    sources: List[SourceCitation]
    retrieved_chunk_count: int
    latency_ms: int
    status: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="Request ID returned by the chat endpoint."
    )
    rating: Literal["up", "down"] = Field(
        ...,
        description="User rating for the chatbot answer."
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional written feedback."
    )


class FeedbackResponse(BaseModel):
    feedback_id: str
    request_id: str
    rating: Literal["up", "down"]
    status: str

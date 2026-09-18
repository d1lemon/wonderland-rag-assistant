import logging
import os
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.auth import get_current_user_id
from app.config import APP_NAME, APP_VERSION
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    SourceCitation,
)
from app.services.persistence_service import (
    create_chat_session,
    get_user_session,
    save_chat_message,
    save_feedback,
)
from app.services.rag_service import WonderlandRAGService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Citation-grounded, authenticated RAG API for "
        "Alice's Adventures in Wonderland."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

rag_service = None


@app.middleware("http")
async def request_id_logging_middleware(
    request: Request,
    call_next,
):
    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = f"req_{uuid.uuid4().hex[:12]}"

    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled API error | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    duration_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)

    logger.info(
        "API request | request_id=%s | method=%s | path=%s | "
        "status=%s | duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.on_event("startup")
def startup_event():
    global rag_service

    raw_groq_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = (
        raw_groq_key
        .replace("\r", "")
        .replace("\n", "")
        .replace(" ", "")
        .strip()
    )

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required but was not found."
        )

    logger.info("Loading RAG service.")

    rag_service = WonderlandRAGService(
        groq_api_key=groq_api_key,
    )

    logger.info("RAG service ready.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/ready")
def readiness_check():
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is still loading.",
        )

    return {
        "status": "ready",
        "chunks_loaded": rag_service.collection.count(),
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    http_request: Request,
    user_id: str = Depends(get_current_user_id),
):
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready.",
        )

    request_id = http_request.state.request_id
    start_time = time.perf_counter()

    if request.session_id:
        session = get_user_session(
            user_id=user_id,
            session_id=request.session_id,
        )

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found.",
            )
    else:
        session = create_chat_session(
            user_id=user_id,
            title=request.question[:80],
        )

    session_id = session["id"]

    user_message = save_chat_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=request.question,
        request_id=request_id,
    )

    logger.info(
        "Chat started | request_id=%s | question_length=%s | top_k=%s",
        request_id,
        len(request.question),
        request.top_k,
    )

    try:
        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
        )

        citations = [
            SourceCitation(
                chunk_id=chunk["chunk_id"],
                chapter_number=chunk["metadata"]["chapter_number"],
                chapter_title=chunk["metadata"]["chapter_title"],
                source_url=chunk["metadata"]["source_url"],
                excerpt=chunk["text"][:300],
                retrieval_distance=round(chunk["distance"], 4),
            )
            for chunk in result["retrieved_chunks"]
        ]

        total_latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        assistant_message = save_chat_message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            request_id=request_id,
            latency_ms=total_latency_ms,
        )

        logger.info(
            "Chat completed | request_id=%s | session_id=%s | "
            "latency_ms=%s | source_count=%s | answer_status=%s",
            request_id,
            session_id,
            total_latency_ms,
            len(citations),
            result["answer_status"],
        )

        return ChatResponse(
            request_id=request_id,
            session_id=session_id,
            user_message_id=user_message["id"],
            assistant_message_id=assistant_message["id"],
            answer=result["answer"],
            sources=citations,
            retrieved_chunk_count=len(citations),
            latency_ms=total_latency_ms,
            retrieval_latency_ms=result["retrieval_latency_ms"],
            generation_latency_ms=result["generation_latency_ms"],
            top_retrieval_distance=result["top_retrieval_distance"],
            answer_status=result["answer_status"],
            status="success",
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Chat failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The chatbot could not process the request. "
                f"Request ID: {request_id}"
            ),
        ) from error


@app.post(
    "/api/v1/feedback",
    response_model=FeedbackResponse,
)
def submit_feedback(
    feedback: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
):
    stored_feedback = save_feedback(
        user_id=user_id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
    )

    return FeedbackResponse(**stored_feedback)

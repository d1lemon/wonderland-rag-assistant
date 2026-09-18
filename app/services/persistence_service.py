from typing import Any

from fastapi import HTTPException, status

from app.supabase_client import supabase_admin


VALID_MESSAGE_ROLES = {"user", "assistant"}
VALID_FEEDBACK_RATINGS = {"up", "down"}


def create_chat_session(
    user_id: str,
    title: str = "New Wonderland Chat",
) -> dict[str, Any]:
    normalized_title = title.strip() or "New Wonderland Chat"

    response = (
        supabase_admin
        .table("chat_sessions")
        .insert(
            {
                "user_id": user_id,
                "title": normalized_title,
            }
        )
        .select("*")
        .execute()
    )

    return response.data[0]


def get_user_session(
    user_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    response = (
        supabase_admin
        .table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None


def require_user_session(
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    session = get_user_session(
        user_id=user_id,
        session_id=session_id,
    )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        )

    return session


def save_chat_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    request_id: str | None = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    normalized_role = role.strip().lower()
    normalized_content = content.strip()

    if normalized_role not in VALID_MESSAGE_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="role must be either 'user' or 'assistant'.",
        )

    if not normalized_content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="content cannot be empty.",
        )

    if latency_ms is not None and latency_ms < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="latency_ms cannot be negative.",
        )

    require_user_session(
        user_id=user_id,
        session_id=session_id,
    )

    response = (
        supabase_admin
        .table("chat_messages")
        .insert(
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": normalized_role,
                "content": normalized_content,
                "request_id": request_id,
                "latency_ms": latency_ms,
            }
        )
        .select("*")
        .execute()
    )

    return response.data[0]


def get_user_assistant_message(
    user_id: str,
    message_id: str,
) -> dict[str, Any] | None:
    response = (
        supabase_admin
        .table("chat_messages")
        .select("*")
        .eq("id", message_id)
        .eq("user_id", user_id)
        .eq("role", "assistant")
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None


def require_user_assistant_message(
    user_id: str,
    message_id: str,
) -> dict[str, Any]:
    message = get_user_assistant_message(
        user_id=user_id,
        message_id=message_id,
    )

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant message not found.",
        )

    return message


def save_feedback(
    user_id: str,
    message_id: str,
    rating: str,
    comment: str | None = None,
) -> dict[str, Any]:
    normalized_rating = rating.strip().lower()
    normalized_comment = comment.strip() if comment else None

    if normalized_rating not in VALID_FEEDBACK_RATINGS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="rating must be either 'up' or 'down'.",
        )

    require_user_assistant_message(
        user_id=user_id,
        message_id=message_id,
    )

    response = (
        supabase_admin
        .table("feedback")
        .upsert(
            {
                "user_id": user_id,
                "message_id": message_id,
                "rating": normalized_rating,
                "comment": normalized_comment,
            },
            on_conflict="message_id,user_id",
        )
        .select("*")
        .execute()
    )

    return response.data[0]
    

import os


APP_NAME = "Wonderland RAG Assistant API"
APP_VERSION = "0.1.0"

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "d1lemon")
REPOSITORY_NAME = "wonderland-rag-assistant"
BRANCH_NAME = "main"

CHUNKS_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USERNAME}/{REPOSITORY_NAME}/{BRANCH_NAME}/"
    f"data/processed/alice_chunks.json"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")

MAX_QUESTION_LENGTH = 500
DEFAULT_TOP_K = 2
MAX_TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.0

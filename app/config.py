import os


APP_NAME = "Wonderland RAG Assistant API"
APP_VERSION = "0.2.0"

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

# Collection uses Chroma's default squared-L2 distance.
# Lower distance means stronger semantic match.
MAX_RETRIEVAL_DISTANCE = float(
    os.getenv("MAX_RETRIEVAL_DISTANCE", "0.65")
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I do not have sufficient support in the provided source material "
    "to answer that."
)

EDUCATIONAL_DISCLAIMER = (
    "Educational information only—not personalized advice."
)

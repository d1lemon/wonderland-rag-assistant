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

# Chroma uses squared-L2 distance for this collection.
# Lower values indicate closer semantic retrieval.
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

# Lightweight input guard for obvious attempts to override instructions,
# request secrets, or refer to hidden model instructions.
INJECTION_BLOCKED_TERMS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the source material",
    "ignore the sources",
    "reveal a secret",
    "reveal the secret",
    "reveal a password",
    "system prompt",
    "developer message",
    "jailbreak",
)

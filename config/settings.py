import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

PLACEHOLDER_MARKER = "buraya_anahtarini_yaz"
DEFAULT_LLM_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
}


def _project_path(name, default):
    path = Path(os.getenv(name, default))
    return str(path if path.is_absolute() else ROOT_DIR / path)


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER not in DEFAULT_LLM_MODELS:
    raise RuntimeError(
        f"LLM_PROVIDER must be one of {list(DEFAULT_LLM_MODELS)}, got '{LLM_PROVIDER}'."
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODELS[LLM_PROVIDER])
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "60"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
DATA_FILE = _project_path("DATA_FILE", "data/raw")
VECTOR_DB_FILE = _project_path("VECTOR_DB_FILE", "vector_db")

ACTIVE_API_KEY_NAME = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}[LLM_PROVIDER]
ACTIVE_API_KEY = GROQ_API_KEY if LLM_PROVIDER == "groq" else OPENAI_API_KEY

if not ACTIVE_API_KEY or PLACEHOLDER_MARKER in ACTIVE_API_KEY:
    raise RuntimeError(
        f"{ACTIVE_API_KEY_NAME} is not set (LLM_PROVIDER={LLM_PROVIDER}). "
        "Fill it in the .env file in the project root."
    )

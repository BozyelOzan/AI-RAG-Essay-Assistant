import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from config.config import GROQ_API_KEY, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE
from src.utils.logger import get_logger

logger = get_logger("ModelSetup")


def setup_models():
    logger.info(f"🔄 Model Setup Process Initiating... (Target LLM: {LLM_MODEL_NAME})")

    try:
        logger.info(f"🧩 Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
        )
        logger.info("✅ Embedding Model loaded into memory.")
        logger.info(f"🧠 Establishing Groq API connection (Model: {LLM_MODEL_NAME})...")
        llm = Groq(
            model=LLM_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.1,
        )

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = CHUNK_SIZE
        logger.info(f"⚙️ Global settings updated (Chunk Size: {CHUNK_SIZE})")
        logger.info("✅ ALL MODELS SUCCESSFULLY LOADED AND CONFIGURED.")

        return llm, embed_model

    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR during model setup: {e}")
        raise e


if __name__ == "__main__":
    setup_models()

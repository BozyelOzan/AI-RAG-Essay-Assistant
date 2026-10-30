from llama_index.core.postprocessor import SentenceTransformerRerank
from config.settings import RERANKER_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger("Retriever")


def get_postprocessors(top_k=3):
    logger.info(f"⚙️ Preparing Reranker... (Target: Top {top_k} results)")

    try:
        logger.info(f"📥 Loading/Downloading '{RERANKER_MODEL_NAME}' model...")

        reranker = SentenceTransformerRerank(model=RERANKER_MODEL_NAME, top_n=top_k)
        logger.info("✅ Reranker model successfully loaded into memory and ready.")
        return [reranker]

    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR while loading Reranker model: {e}")
        raise e

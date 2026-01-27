from llama_index.core.postprocessor import SentenceTransformerRerank
from src.utils.logger import get_logger

logger = get_logger("Retriever")


def get_postprocessors(top_k=3):
    logger.info(f"⚙️ Preparing Reranker... (Target: Top {top_k} results)")

    try:
        logger.info("📥 Loading/Downloading 'BAAI/bge-reranker-base' model...")

        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", top_n=top_k
        )
        logger.info("✅ Reranker model successfully loaded into memory and ready.")
        return [reranker]

    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR while loading Reranker model: {e}")
        raise e

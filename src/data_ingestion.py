import sys
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import setup_models
from config.config import DATA_FILE, VECTOR_DB_FILE

from src.utils.logger import get_logger

logger = get_logger("DataIngestion")


def load_data():
    logger.info("🎬 Data Ingestion Process Started...")

    try:
        setup_models()
        logger.info("✅ LLM and Embedding models successfully configured.")
    except Exception as e:
        logger.error(f"❌ Error in model setup: {e}")
        raise e

    logger.info(f"📂 Data Source: {DATA_FILE}")
    logger.info(f"🗄️ Vector Database Target: {VECTOR_DB_FILE}")

    try:
        db = chromadb.PersistentClient(path=VECTOR_DB_FILE)
        chroma_collection = db.get_or_create_collection("rag_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        sys.exit(1)

    if chroma_collection.count() > 0:
        logger.info(
            f"✅ Database already populated ({chroma_collection.count()} chunks exist). Loading existing index..."
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        logger.info("🚀 Database is empty. Starting new indexing process...")

        if os.path.exists(DATA_FILE):
            files = [f for f in os.listdir(DATA_FILE) if f.endswith(".pdf")]
            if not files:
                logger.warning(
                    f"⚠️ WARNING: Directory '{DATA_FILE}' is empty! No PDFs found to index."
                )
                return None

            logger.info(f"📄 Found {len(files)} PDF files in directory:")
            for f in files:
                logger.info(f"  -> Found File: {f}")
        else:
            error_msg = f"❌ ERROR: Directory '{DATA_FILE}' not found!"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("📖 Reading PDF files (Parsing)...")
        try:
            documents = SimpleDirectoryReader(DATA_FILE).load_data()
            logger.info(
                f"✅ Parsing Successful: A total of {len(documents)} document pages loaded into memory."
            )
        except Exception as e:
            logger.error(f"❌ Error occurred while reading PDFs: {e}")
            raise e

        logger.info(
            "🧠 Starting Vectorization and Embedding generation (This process may take time)..."
        )
        try:
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, show_progress=True
            )
            logger.info("💾 Indexing completed and saved to disk.")
        except Exception as e:
            logger.error(f"❌ Error during vectorization: {e}")
            raise e

    return index


if __name__ == "__main__":
    load_data()

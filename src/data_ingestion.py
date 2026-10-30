import sys
import os
import chromadb
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_setup import setup_models
from config.settings import DATA_FILE, VECTOR_DB_FILE

from src.utils.logger import get_logger

logger = get_logger("DataIngestion")

COLLECTION_NAME = "rag_collection"


def _find_pdfs():
    if not os.path.isdir(DATA_FILE):
        logger.warning(f"⚠️ Directory '{DATA_FILE}' not found.")
        return []

    return sorted(
        os.path.join(DATA_FILE, f)
        for f in os.listdir(DATA_FILE)
        if f.lower().endswith(".pdf")
    )


def _read_pdfs(pdf_paths):
    logger.info(f"📖 Reading {len(pdf_paths)} PDF file(s) (Parsing)...")
    try:
        documents = SimpleDirectoryReader(input_files=pdf_paths).load_data()
        logger.info(
            f"✅ Parsing Successful: A total of {len(documents)} document pages loaded into memory."
        )
        return documents
    except Exception as e:
        logger.error(f"❌ Error occurred while reading PDFs: {e}")
        raise e


def _indexed_file_names(collection):
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return {meta.get("file_name") for meta in metadatas if meta}


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
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        raise e

    pdf_paths = _find_pdfs()
    logger.info(f"📄 Found {len(pdf_paths)} PDF file(s) in directory.")

    if chroma_collection.count() == 0:
        if not pdf_paths:
            logger.warning("⚠️ Database is empty and no PDFs were found to index.")
            return None

        logger.info("🚀 Database is empty. Starting new indexing process...")
        documents = _read_pdfs(pdf_paths)
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

    logger.info(
        f"✅ Database already populated ({chroma_collection.count()} chunks exist). Loading existing index..."
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    indexed_files = _indexed_file_names(chroma_collection)
    new_pdfs = [p for p in pdf_paths if os.path.basename(p) not in indexed_files]
    if not new_pdfs:
        logger.info("✅ Index is up to date. No new PDFs to add.")
        return index

    logger.info(f"🆕 {len(new_pdfs)} new PDF file(s) found. Adding them to the index...")
    documents = _read_pdfs(new_pdfs)
    try:
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        index.insert_nodes(nodes)
        logger.info(f"💾 {len(nodes)} new chunks added to the database.")
    except Exception as e:
        logger.error(f"❌ Error while adding new PDFs to the index: {e}")
        raise e

    return index


if __name__ == "__main__":
    load_data()

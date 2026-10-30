import sys
import os
from llama_index.core.memory import ChatMemoryBuffer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data
from src.prompts import SYSTEM_PROMPT
from src.retriever import get_postprocessors

from config.settings import RERANK_TOP_N, SIMILARITY_TOP_K
from src.utils.logger import get_logger

logger = get_logger("RAGEngine")


def create_chat_engine(index, node_postprocessors):
    logger.info("🧠 Setting up chat memory (Memory Buffer) (Limit: 3000 tokens)...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    logger.info(
        f"🚀 Engine configuration: Top-K={SIMILARITY_TOP_K} (Wide Search) + Reranker (Top {RERANK_TOP_N})"
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        similarity_top_k=SIMILARITY_TOP_K,
        node_postprocessors=node_postprocessors,
        verbose=True,
    )

    logger.info("✅ Chat Engine successfully created and ready for use.")
    return chat_engine


if __name__ == "__main__":
    logger.info("🤖 Starting Enterprise RAG Chatbot in CLI Mode...")

    try:
        index = load_data()
        if index is None:
            logger.critical("❌ No index available. Add PDF files to the data folder.")
            sys.exit(1)

        engine = create_chat_engine(index, get_postprocessors(top_k=RERANK_TOP_N))

        while True:
            soru = input("You: ")
            if soru.lower() in ["q", "exit", "quit"]:
                logger.info("🛑 User requested exit. Shutting down system.")
                break
            logger.info(f"📨 User Question Received: '{soru}'")

            try:
                response = engine.chat(soru)
            except Exception as e:
                logger.error(f"❌ Error generating response: {e}")
                continue

            logger.info(f"📤 AI Response Generated: {str(response)[:100]}...")
            log_message = "🔍 Sources used for this response:\n"

            for i, node in enumerate(response.source_nodes):
                file_name = node.metadata.get("file_name", "Unknown")
                score = node.score or 0.0
                log_message += f"   [{i + 1}] Score: {score:.4f} | File: {file_name}\n"

            logger.info(log_message.strip())

    except Exception as e:
        logger.critical(f"❌ System Runtime Error: {e}")

import sys
import os
from llama_index.core.memory import ChatMemoryBuffer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data
from src.prompts import SYSTEM_PROMPT
from src.retriever import get_postprocessors

from src.utils.logger import get_logger

logger = get_logger("RAGEngine")


def get_chat_engine():
    logger.info("⚙️ Preparing RAG Engine (Chat Engine)...")

    try:
        index = load_data()
        logger.info("🧠 Setting up chat memory (Memory Buffer) (Limit: 3000 tokens)...")
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        logger.info(
            "🚀 Engine configuration: Top-K=60 (Wide Search) + Reranker (Smart Filter)"
        )

        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=60,
            node_postprocessors=get_postprocessors(top_k=5),
            verbose=True,
        )

        logger.info("✅ Chat Engine successfully created and ready for use.")
        return chat_engine

    except Exception as e:
        logger.error(f"❌ Critical error while creating Chat Engine: {e}")
        raise e


if __name__ == "__main__":
    logger.info("🤖 Starting Enterprise RAG Chatbot in CLI Mode...")

    try:
        engine = get_chat_engine()

        while True:
            soru = input("You: ")
            if soru.lower() in ["q", "exit", "quit"]:
                logger.info("🛑 User requested exit. Shutting down system.")
                break
            logger.info(f"📨 User Question Received: '{soru}'")
            response = engine.chat(soru)
            logger.info(f"📤 AI Response Generated: {str(response)[:100]}...")
            log_message = "🔍 Sources used for this response:\n"

            for i, node in enumerate(response.source_nodes):
                file_name = node.metadata.get("file_name", "Unknown")
                score = node.score
                log_message += f"   [{i + 1}] Score: {score:.4f} | File: {file_name}\n"

            logger.info(log_message.strip())

    except Exception as e:
        logger.critical(f"❌ System Runtime Error: {e}")

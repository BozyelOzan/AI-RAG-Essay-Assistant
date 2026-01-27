import streamlit as st
import os

from src.utils.logger import get_logger

logger = get_logger("StreamlitApp")

from src.engine import get_chat_engine

st.set_page_config(
    page_title="AI Research Assistant", page_icon="🧬", layout="centered"
)


def main():
    st.title("🧬 AI Research Assistant")
    st.markdown("""
    This assistant analyzes the **Computer Vision and Deep Learning** papers you upload.
                It operates using a *Retrieval-Augmented Generation (RAG) architecture.
    """)

    with st.sidebar:
        st.header("📂 **Document Status**")
        try:
            if not os.path.exists("data/raw"):
                os.makedirs("data/raw")

            files = os.listdir("data/raw")
            if files:
                st.success(f"{len(files)} articles have been indexed.")
                st.markdown("**Uploaded Papers:**")
                for f in files:
                    st.code(f, language="text")
            else:
                st.error("No documents have been uploaded yet!")
                logger.warning(
                    "⚠️ Sidebar: The `data/raw` directory appears to be empty."
                )
        except Exception as e:
            logger.error(f"❌ Error while loading the sidebar: {e}")
            st.error("Failed to load the file list.")

        st.divider()
        st.info("Model: Llama-3.3-70b (Groq)\nEmbedding: all-MiniLM-L6-v2")

    if "chat_engine" not in st.session_state:
        logger.info("⚙️ Chat Engine (Session) is starting...")

        try:
            with st.spinner("Engine is starting..."):
                st.session_state.chat_engine = get_chat_engine()
                logger.info("✅ Chat Engine successfully loaded into session_state.")
        except Exception as e:
            logger.critical(f"❌ Engine failed to start: {e}")
            st.error(
                "A critical error occurred during system initialization. Please check the logs."
            )
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your technical document assistant. You can ask me questions about your articles.",
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        logger.info(f"👤 User Question: {prompt}")

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Scanning documents..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response)
                    logger.info("🤖 AI response successfully generated and displayed.")

                    with st.expander("📚 Sources Used (Evidence)"):
                        for node in response.source_nodes:
                            metadata = node.metadata
                            file_name = metadata.get("file_name", "Unknown")
                            page_label = metadata.get("page_label", "?")
                            score = node.score if node.score else 0.0

                            st.markdown(
                                f"**📄 File:** `{file_name}` | **Page:** `{page_label}` | **Score:** `{score:.4f}`"
                            )
                            st.info(f"...{node.node.get_content()[:300]}...")
                            st.divider()

                except Exception as e:
                    logger.error(f"❌ Error generating response: {e}")
                    st.error("An error occurred. Please try again.")

        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )


if __name__ == "__main__":
    main()

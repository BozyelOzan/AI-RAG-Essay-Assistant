import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core.postprocessor import SentenceTransformerRerank

from src.data_ingestion import load_data
from config.settings import RERANKER_MODEL_NAME, SIMILARITY_TOP_K


def _one_line(text):
    return text.replace("\n", " ")


def debug_search(query):
    print(f"\n🔍 SEARCH TERM: '{query}'")
    print("=" * 50)

    index = load_data()
    if index is None:
        print("❌ ERROR: No index available. Add PDF files to the data folder.")
        return

    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)

    print(f"\n--- PHASE 1: Vector Search Results ({len(nodes)} items) ---")
    for i, node in enumerate(nodes):
        score = node.score or 0.0
        print(
            f"[{i + 1}] Score: {score:.4f} | File: {node.metadata.get('file_name')}"
        )
        preview = _one_line(node.get_content())[:150]
        print(f"    Content Preview: {preview}...")
        print("-" * 20)

    print("\n--- PHASE 2: After Reranker (Cross-Encoder) ---")
    reranker = SentenceTransformerRerank(model=RERANKER_MODEL_NAME, top_n=30)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)

    if not reranked_nodes:
        print(
            "❌ ERROR: Reranker filtered out all results! (All fell below the threshold)"
        )
    else:
        for i, node in enumerate(reranked_nodes):
            score = node.score or 0.0
            print(
                f"[{i + 1}] Score: {score:.4f} | File: {node.metadata.get('file_name')}"
            )
            content = _one_line(node.get_content())[:150]
            print(f"    Content: {content}...")


if __name__ == "__main__":
    debug_search("U-Net model architecture")

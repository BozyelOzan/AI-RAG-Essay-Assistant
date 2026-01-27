import sys
import os
from llama_index.core.postprocessor import SentenceTransformerRerank

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_ingestion import load_data


def debug_search(query):
    print(f"\n🔍 SEARCH TERM: '{query}'")
    print("=" * 50)

    index = load_data()
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)

    print(f"\n--- PHASE 1: Vector Search Results ({len(nodes)} items) ---")
    for i, node in enumerate(nodes):
        print(
            f"[{i + 1}] Score: {node.score:.4f} | File: {node.metadata.get('file_name')}"
        )
        print(f"    Content Preview: {node.get_content().replace('\n', ' ')[:150]}...")
        print("-" * 20)

    print("\n--- PHASE 2: After Reranker (Cross-Encoder) ---")
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=30)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)

    if not reranked_nodes:
        print(
            "❌ ERROR: Reranker filtered out all results! (All fell below the threshold)"
        )
    else:
        for i, node in enumerate(reranked_nodes):
            print(
                f"[{i + 1}] Score: {node.score:.4f} | File: {node.metadata.get('file_name')}"
            )
            print(f"    Content: {node.get_content().replace('\n', ' ')[:150]}...")


if __name__ == "__main__":
    debug_search("U-Net model architecture")

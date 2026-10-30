import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from config.settings import VECTOR_DB_FILE

COLLECTION_NAME = "rag_collection"
UNET_ARXIV_ID = "1505.04597"


def inspect_db():
    print(f"📂 Database Path: {VECTOR_DB_FILE}")

    db = chromadb.PersistentClient(path=VECTOR_DB_FILE)
    try:
        collection = db.get_collection(COLLECTION_NAME)
    except Exception:
        print("❌ Collection not found! Database might be corrupt or empty.")
        return

    count = collection.count()
    print(f"📊 Total Chunk Count: {count}")
    result = collection.get(include=["metadatas", "documents"])

    unique_files = set()
    unet_chunks = []

    print("\n🔍 Scanning Files in Database...")
    for i, meta in enumerate(result["metadatas"]):
        file_name = meta.get("file_name", "UNKNOWN")
        unique_files.add(file_name)
        if UNET_ARXIV_ID in file_name or "unet" in file_name.lower():
            unet_chunks.append(result["documents"][i])

    print("\n📄 Unique Files Found:")
    for f in sorted(unique_files):
        print(f"  - {f}")

    print("-" * 30)

    if unet_chunks:
        print(f"✅ U-NET FILE FOUND! ({len(unet_chunks)} chunks)")
        print("\n--- CONTENT PREVIEW (First chunk) ---")
        print(unet_chunks[0][:500])
        print("-----------------------------------")
        found_keyword = any("U-Net" in chunk for chunk in unet_chunks)
        if found_keyword:
            print("✅ The keyword 'U-Net' appears exactly in the text.")
        else:
            print(
                "⚠️ WARNING: File exists but the exact string 'U-Net' was not found in the content!"
            )
            print("   (It was likely read incorrectly as 'U - Net' or 'UNet'.)")
    else:
        print("❌ CRITICAL ERROR: U-Net file NOT found in database!")
        print(
            "   (The file may have been skipped or failed to save during data_ingestion.)"
        )


if __name__ == "__main__":
    inspect_db()

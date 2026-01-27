import sys
import os
import chromadb
from config.config import VECTOR_DB_FILE


def inspect_db():
    print(f"📂 Database Path: {VECTOR_DB_FILE}")

    db = chromadb.PersistentClient(path=VECTOR_DB_FILE)
    try:
        collection = db.get_collection("rag_collection")
    except Exception as e:
        print("❌ Collection not found! Database might be corrupt or empty.")
        return

    count = collection.count()
    print(f"📊 Total Chunk Count: {count}")
    result = collection.get(include=["metadatas", "documents"])

    unique_files = set()
    vnet_chunks = []

    print("\n🔍 Scanning Files in Database...")
    for i, meta in enumerate(result["metadatas"]):
        file_name = meta.get("file_name", "UNKNOWN")
        unique_files.add(file_name)
        if "1606.04797" in file_name or "vnet" in file_name.lower():
            vnet_chunks.append(result["documents"][i])

    print("\n📄 Unique Files Found:")
    for f in sorted(list(unique_files)):
        print(f"  - {f}")

    print("-" * 30)

    if vnet_chunks:
        print(f"✅ U-NET FILE FOUND! ({len(vnet_chunks)} chunks)")
        print("\n--- CONTENT PREVIEW (First chunk) ---")
        print(vnet_chunks[0][:500])
        print("-----------------------------------")
        found_keyword = any("U-Net" in chunk for chunk in vnet_chunks)
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

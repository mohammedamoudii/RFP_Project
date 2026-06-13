from pathlib import Path
from collections import Counter, defaultdict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


RFP_DB_PATH = Path("data/chroma/rfp_db")
RFP_COLLECTION = "rfp_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=str(RFP_DB_PATH))
    collection = client.get_collection(
        name=RFP_COLLECTION,
        embedding_function=embedding_function,
    )

    data = collection.get(include=["metadatas"])

    metadatas = data.get("metadatas", [])

    opportunity_counter = Counter()
    folder_names = defaultdict(set)

    for metadata in metadatas:
        opportunity_id = metadata.get("opportunity_id", "")
        folder_name = metadata.get("folder_name", "")

        if opportunity_id:
            opportunity_counter[opportunity_id] += 1

        if opportunity_id and folder_name:
            folder_names[opportunity_id].add(folder_name)

    print(f"Total RFP chunks: {collection.count()}")
    print(f"Unique RFP opportunities: {len(opportunity_counter)}")
    print("\nTop opportunities by chunk count:\n")

    for opportunity_id, count in opportunity_counter.most_common():
        names = ", ".join(sorted(folder_names[opportunity_id]))
        print(f"{opportunity_id} | chunks: {count} | folder: {names}")


if __name__ == "__main__":
    main()
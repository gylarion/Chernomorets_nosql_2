import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Індекс {INDEX_NAME} створено")
else:
    print(f"Індекс {INDEX_NAME} вже існує")

index = pc.Index(INDEX_NAME)

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Завантажуємо {len(df)} векторів у Pinecone...")

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    batch_emb = embeddings[i:i+BATCH_SIZE]

    vectors = []
    for j, (_, row) in enumerate(batch_df.iterrows()):
        vectors.append({
            "id": f"paper_{i+j}",
            "values": batch_emb[j].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"])
            }
        })

    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(f"\nВекторів в індексі: {stats.total_vector_count}")
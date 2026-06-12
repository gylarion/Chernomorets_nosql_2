import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
CHUNK_SIZE = 50
OVERLAP = 10
MAX_WORDS = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

top30 = df.assign(abstract_len=df["abstract"].str.len()).nlargest(30, "abstract_len").drop(columns="abstract_len").copy()

def fixed_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def semantic_chunk(text: str, max_words: int = MAX_WORDS) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks

def create_index(name):
    if name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Індекс {name} створено")
    else:
        print(f"Індекс {name} вже існує")
    return pc.Index(name)

fixed_index = create_index(FIXED_INDEX)
semantic_index = create_index(SEMANTIC_INDEX)

def upload_chunks(index, chunks_data, batch_size=100):
    for i in tqdm(range(0, len(chunks_data), batch_size)):
        batch = chunks_data[i:i+batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True)
        vectors = [{
            "id": c["id"],
            "values": embeddings[j].tolist(),
            "metadata": {
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "chunk_text": c["text"][:500],
                "chunk_num": c["chunk_num"],
                "year": c["year"],
                "category": c["category"]
            }
        } for j, c in enumerate(batch)]
        index.upsert(vectors=vectors)

print("Готуємо fixed chunks...")
fixed_chunks = []
for _, row in top30.iterrows():
    for i, chunk in enumerate(fixed_chunk(row["abstract"])):
        fixed_chunks.append({
            "id": f"fixed_{row['id']}_{i}",
            "arxiv_id": row["id"],
            "title": row["title"],
            "text": chunk,
            "chunk_num": i,
            "year": int(row["year"]),
            "category": row["category"]
        })

print(f"Fixed chunks: {len(fixed_chunks)}")
upload_chunks(fixed_index, fixed_chunks)

print("\nГотуємо semantic chunks...")
semantic_chunks = []
for _, row in top30.iterrows():
    for i, chunk in enumerate(semantic_chunk(row["abstract"])):
        semantic_chunks.append({
            "id": f"semantic_{row['id']}_{i}",
            "arxiv_id": row["id"],
            "title": row["title"],
            "text": chunk,
            "chunk_num": i,
            "year": int(row["year"]),
            "category": row["category"]
        })

print(f"Semantic chunks: {len(semantic_chunks)}")
upload_chunks(semantic_index, semantic_chunks)

def search_chunks(index, query, top_k=5, label=""):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    print(f"\n{'='*60}")
    print(f"{label}: '{query}'")
    print(f"{'='*60}")
    for i, match in enumerate(results["matches"]):
        print(f"\n#{i+1} score={match['score']:.4f}")
        print(f"Title: {match['metadata']['title'][:70]}")
        print(f"Chunk #{match['metadata']['chunk_num']}: {match['metadata']['chunk_text'][:200]}")

test_queries = [
    "quantum mechanics energy levels",
    "statistical methods data analysis"
]

for q in test_queries:
    search_chunks(fixed_index, q, label="Fixed chunks")
    search_chunks(semantic_index, q, label="Semantic chunks")

print(f"\nFixed chunks в індексі: {fixed_index.describe_index_stats().total_vector_count}")
print(f"Semantic chunks в індексі: {semantic_index.describe_index_stats().total_vector_count}")
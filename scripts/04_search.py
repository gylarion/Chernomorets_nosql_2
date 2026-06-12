import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

def encode_query(query: str) -> list:
    return model.encode(query, normalize_embeddings=True).tolist()

def print_results(results, label=""):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    for i, match in enumerate(results["matches"]):
        print(f"\n#{i+1} score={match['score']:.4f}")
        print(f"Title: {match['metadata']['title']}")
        print(f"Category: {match['metadata']['category']} | Year: {match['metadata']['year']}")
        print(f"Abstract: {match['metadata']['abstract'][:200]}...")

query1 = "teaching machines to recognize objects in pictures"
vec1 = encode_query(query1)
results1 = index.query(vector=vec1, top_k=TOP_K, include_metadata=True)
print_results(results1, f"Семантичний пошук: '{query1}'")

query2 = "reinforcement learning neural networks"
vec2 = encode_query(query2)
results2a = index.query(
    vector=vec2,
    top_k=TOP_K,
    include_metadata=True,
    filter={"category": {"$eq": "cs.LG"}, "year": {"$gte": 2003}}
)
print_results(results2a, f"Фільтр A: reinforcement learning, категорія cs.LG, після 2003")

results2b = index.query(
    vector=vec2,
    top_k=TOP_K,
    include_metadata=True,
    filter={"year": {"$lte": 2007}}
)
print_results(results2b, f"Фільтр B: до 2007 року")

print(f"\n{'='*60}")
print("Порівняння метрик схожості")
print(f"{'='*60}")

all_embeddings = np.load("embeddings/embeddings.npy")
query_vec = np.array(encode_query("deep learning image classification"))

cosine_scores = all_embeddings @ query_vec
dot_scores = all_embeddings @ query_vec
l2_scores = -np.linalg.norm(all_embeddings - query_vec, axis=1)

for label, scores in [("Cosine", cosine_scores), ("Dot Product", dot_scores), ("L2 (негативна)", l2_scores)]:
    top5 = np.argsort(scores)[::-1][:5]
    print(f"\n--- {label} ---")
    for rank, idx in enumerate(top5):
        print(f"#{rank+1} [{df.iloc[idx]['category']}] {df.iloc[idx]['title'][:80]}")
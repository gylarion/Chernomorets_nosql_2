import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

def bm25_search(query: str, top_k: int = TOP_K) -> list:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_idx]

def vector_search(query: str, top_k: int = TOP_K) -> list:
    vec = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    out = []
    for match in results["matches"]:
        paper_id = int(match["id"].replace("paper_", ""))
        out.append((paper_id, float(match["score"])))
    return out

def rrf(bm25_results: list, vector_results: list, k: int = 60) -> list:
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def print_bm25(results, label):
    print(f"\n{'='*60}")
    print(f"BM25 | {label}")
    print(f"{'='*60}")
    for rank, (idx, score) in enumerate(results[:5]):
        row = df.iloc[idx]
        print(f"#{rank+1} score={score:.4f} [{row['category']}] {row['title'][:70]}")

def print_vector(results, label):
    print(f"\n{'='*60}")
    print(f"Vector | {label}")
    print(f"{'='*60}")
    for rank, (idx, score) in enumerate(results[:5]):
        row = df.iloc[idx]
        print(f"#{rank+1} score={score:.4f} [{row['category']}] {row['title'][:70]}")

def print_hybrid(results, label):
    print(f"\n{'='*60}")
    print(f"Hybrid RRF | {label}")
    print(f"{'='*60}")
    for rank, (idx, score) in enumerate(results[:5]):
        row = df.iloc[idx]
        print(f"#{rank+1} rrf={score:.6f} [{row['category']}] {row['title'][:70]}")

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for query in queries:
    print(f"\n{'#'*60}")
    print(f"ЗАПИТ: {query}")
    print(f"{'#'*60}")

    bm25_res = bm25_search(query)
    vector_res = vector_search(query)
    hybrid_res = rrf(bm25_res, vector_res)

    print_bm25(bm25_res, query)
    print_vector(vector_res, query)
    print_hybrid(hybrid_res, query)
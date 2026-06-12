import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_FILE)

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

print(f"Завантажуємо модель {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Генеруємо ембеддинги для {len(texts)} текстів...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"\nКількість текстів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

np.save(OUTPUT_FILE, embeddings)
print(f"\nЗбережено у {OUTPUT_FILE}")
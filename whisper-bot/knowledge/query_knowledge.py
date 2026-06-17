import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np
from MiniLM.knowledge_inference import get_embedding
import os
DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_db (4).pkl")


# Load or initialize the knowledge database
try:
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
except FileNotFoundError:
    db = {"chunks": [], "embeddings": []}

chunks = db["chunks"]
embeddings = np.array(db["embeddings"], dtype=np.float32) if len(db["embeddings"]) > 0 else np.empty((0, 0), dtype=np.float32)


def cosine_similarity(a, b):
    return np.dot(a, b)


def save_db():
    """Persist the current chunks and embeddings to disk."""
    # Convert embeddings to list for pickle compatibility
    db["chunks"] = chunks
    db["embeddings"] = embeddings.tolist()
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

import re

while True:
    question = input("\nQuestion: ")
    if question.lower() == "exit":
        break

    # Training command pattern: train on <full_sentence>
    train_match = re.match(r'^train on (.+)$', question, re.IGNORECASE)
    if train_match:
        full_sentence = train_match.group(1).strip()
        # Use the full sentence as the response text
        response_text = full_sentence
        # Generate embedding for the sentence
        new_emb = get_embedding(full_sentence)
        # Append new entry to existing knowledge base
        if embeddings.size == 0:
            embeddings = np.expand_dims(new_emb, axis=0)
        else:
            embeddings = np.vstack([embeddings, new_emb])
        if not isinstance(chunks, list):
            chunks = []
        chunks.append(response_text)
        save_db()
        print(f"Trained on: '{full_sentence}'")
        continue

    query_embedding = get_embedding(question)
    scores = embeddings @ query_embedding
    best_idx = np.argmax(scores)
    print("\nSimilarity:", round(float(scores[best_idx]), 3))
    print("\nAnswer:\n")
    print(chunks[best_idx])
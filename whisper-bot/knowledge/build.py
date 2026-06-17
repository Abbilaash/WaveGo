import pickle
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_PATH = "/content/whisper_knowledge (3).pdf"
OUTPUT_PATH = "/content/knowledge_db.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text, size=500, overlap=100):

    words = text.split()

    '''chunks = []

    start = 0

    while start < len(words):

        end = start + size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += size - overlap

    return chunks'''
    chunks = []

    for line in full_text.split("\n"):
        line = line.strip()

        if len(line) < 10:
            continue

        chunks.append(line)

    print("Number of chunks:", len(chunks))
    return chunks


print("Loading PDF...")

reader = PdfReader(PDF_PATH)

full_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        full_text += text + "\n"


print("Chunking...")

chunks = chunk_text(
    full_text,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

print(f"Created {len(chunks)} chunks")

print("Loading MiniLM...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

database = {
    "chunks": chunks,
    "embeddings": embeddings.astype(np.float32)
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(database, f)

print(f"Saved to {OUTPUT_PATH}")
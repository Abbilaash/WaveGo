#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import numpy as np
import pypdf

# Add parent directory of 'knowledge' (whisper-bot) to sys.path so we can import MiniLM
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

try:
    from MiniLM.knowledge_inference import get_embedding
except ImportError as e:
    print(f"Error: Could not import MiniLM.knowledge_inference. Make sure {base_dir} is in sys.path. Details: {e}")
    sys.exit(1)

def split_text_clean(text, chunk_size=800, overlap=150):
    """
    Split text into chunks of target character length, keeping word boundaries.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1  # +1 for space
        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Create overlap: grab last words that fit overlap size
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk):
                if overlap_len + len(w) + 1 > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current_chunk = overlap_words
            current_len = overlap_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF and generate vector database using MiniLM.")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", default=None, help="Path to the output pickle file. Defaults to knowledge_db.pkl in the same directory.")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    # Resolve output path
    if args.output is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_db.pkl")
    else:
        output_path = os.path.abspath(args.output)

    print(f"Reading PDF from: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
        full_text = "\n".join(pages_text)
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        sys.exit(1)

    if not full_text.strip():
        print("Error: No text extracted from PDF.")
        sys.exit(1)

    print(f"Extracted text from {len(reader.pages)} pages. Total length: {len(full_text)} characters.")
    
    chunks = split_text_clean(full_text)
    print(f"Split text into {len(chunks)} chunks.")

    print("Generating embeddings using MiniLM...")
    embeddings = []
    for idx, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb.tolist())  # Save as standard list of floats
            if (idx + 1) % 10 == 0 or (idx + 1) == len(chunks):
                print(f" Progress: {idx + 1}/{len(chunks)} chunks processed.")
        except Exception as e:
            print(f"Error generating embedding for chunk {idx}: {e}")
            sys.exit(1)

    # Save to file
    db_data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    
    print(f"Saving vector database to: {output_path}")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(db_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Vector database created successfully!")
    except Exception as e:
        print(f"Error saving vector database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

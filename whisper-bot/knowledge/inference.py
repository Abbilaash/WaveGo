#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Add parent directory of 'knowledge' (whisper-bot) to sys.path so we can import MiniLM
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

try:
    from MiniLM.knowledge_inference import get_embedding
except ImportError as e:
    print(f"Error: Could not import MiniLM.knowledge_inference. Make sure {base_dir} is in sys.path. Details: {e}")
    sys.exit(1)

def run_inference(query, db_path, model_path, tokenizer_path, top_k=3, max_new_tokens=256, temperature=0.0):
    # --------------------------------
    # 1. Load Vector Database
    # --------------------------------
    if not os.path.exists(db_path):
        print(f"Error: Vector database not found at '{db_path}'.")
        print("Please generate it first by running:")
        print(f"  python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_db_gen.py')} --pdf <your_pdf_path>")
        sys.exit(1)

    print(f"Loading vector database from: {db_path}")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    chunks = db.get("chunks", [])
    embeddings = db.get("embeddings", [])
    
    if not chunks or not embeddings:
        print("Error: Vector database is empty or has invalid format.")
        sys.exit(1)

    # --------------------------------
    # 2. Vector Search (Retrieve Context)
    # --------------------------------
    print(f"Generating query embedding...")
    query_emb = get_embedding(query)
    
    print(f"Searching database (top-{top_k} similarity)...")
    similarities = []
    for emb in embeddings:
        # Cosine similarity (since embeddings are L2 normalized, dot product is cosine similarity)
        sim = np.dot(query_emb, np.array(emb))
        similarities.append(sim)

    # Get top_k indices sorted descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print("\n--- Retrieved Context Chunks ---")
    retrieved_chunks = []
    for idx in top_indices:
        score = similarities[idx]
        chunk_text = chunks[idx]
        print(f"[Score: {score:.4f}] {chunk_text[:120]}...")
        retrieved_chunks.append(chunk_text)
    print("--------------------------------\n")

    context = "\n\n".join(retrieved_chunks)

    # --------------------------------
    # 3. Construct Gemma3 Chat Prompt
    # --------------------------------
    # Note: Tokenizer prepends BOS (<bos>) automatically, so we construct from start_of_turn
    prompt = f"<start_of_turn>user\nContext:\n{context}\n\nQuestion:\n{query}\n\nInstructions: Answer the question concisely in 1 sentence.<end_of_turn>\n<start_of_turn>model\n"

    # --------------------------------
    # 4. Initialize Gemma3 ONNX Model & Tokenizer
    # --------------------------------
    if not os.path.exists(tokenizer_path):
        print(f"Error: Gemma3 tokenizer.json not found at '{tokenizer_path}'.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Gemma3 model not found at '{model_path}'.")
        sys.exit(1)

    print("Loading Gemma3 tokenizer...")
    gemma_tokenizer = Tokenizer.from_file(tokenizer_path)

    print("Initializing Gemma3 ONNX Session...")
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print("\n" + "="*80)
        print("CRITICAL ERROR: Failed to load Gemma3 ONNX model.")
        print(f"Details: {e}")
        if "model.onnx_data" in str(e) or "external data" in str(e).lower():
            print("\nEXPLANATION:")
            print("The model 'gemma3.onnx' references external weight data stored in 'model.onnx_data'.")
            print("Currently, 'model.onnx_data' is missing from the directory.")
            print("\nHOW TO FIX:")
            print(f"Please obtain and place 'model.onnx_data' into: {os.path.dirname(model_path)}")
        print("="*80 + "\n")
        sys.exit(1)

    # --------------------------------
    # 5. Token Generation Loop with KV Cache
    # --------------------------------
    print("\nGenerating response: ", end="")
    sys.stdout.flush()

    response = generate_response(
        session=session,
        tokenizer=gemma_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        print_stream=True
    )
    print()

def generate_response(session, tokenizer, prompt, max_new_tokens=256, temperature=0.0, print_stream=False):
    encoding = tokenizer.encode(prompt)
    input_ids = encoding.ids
    
    batch_size = 1
    seq_len = len(input_ids)
    
    # Initialize past key values (KV Cache) for all 18 layers with shape (batch_size, 1, 0, 256)
    past_key_values = {}
    for i in range(18):
        past_key_values[f"past_key_values.{i}.key"] = np.zeros((batch_size, 1, 0, 256), dtype=np.float32)
        past_key_values[f"past_key_values.{i}.value"] = np.zeros((batch_size, 1, 0, 256), dtype=np.float32)

    generated_tokens = []
    curr_input_ids = np.array([input_ids], dtype=np.int64)
    curr_attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    
    # Define outputs we want to retrieve
    output_names = ["logits"]
    for i in range(18):
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    last_text = ""

    for step in range(max_new_tokens):
        # Prepare inputs feed
        inputs = {
            "input_ids": curr_input_ids,
            "attention_mask": curr_attention_mask
        }
        inputs.update(past_key_values)
        
        # Run inference
        try:
            outputs = session.run(output_names, inputs)
        except Exception as e:
            if print_stream:
                print(f"\nError running model generation at step {step}: {e}")
            break

        logits = outputs[0]
        # Get logits of the last token
        next_token_logits = logits[0, -1, :]

        # Sampling logic
        if temperature == 0.0:
            next_token = int(np.argmax(next_token_logits))
        else:
            # apply temperature
            logits_scaled = next_token_logits / temperature
            # stable softmax
            exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
            probs = exp_logits / np.sum(exp_logits)
            next_token = int(np.random.choice(len(probs), p=probs))

        generated_tokens.append(next_token)
        
        # Check for EOS token
        if next_token in [1, 106]:  # 1 is <eos>, 106 is <end_of_turn>
            break

        # Stream token decoding
        current_text = tokenizer.decode(generated_tokens)
        if print_stream:
            new_text = current_text[len(last_text):]
            sys.stdout.write(new_text)
            sys.stdout.flush()
        last_text = current_text

        # Update KV cache for next iteration
        for i in range(18):
            past_key_values[f"past_key_values.{i}.key"] = outputs[1 + 2*i]
            past_key_values[f"past_key_values.{i}.value"] = outputs[1 + 2*i + 1]

        # Set input to only the newly generated token
        curr_input_ids = np.array([[next_token]], dtype=np.int64)
        past_seq_len = past_key_values["past_key_values.0.key"].shape[2]
        curr_attention_mask = np.ones((batch_size, past_seq_len + 1), dtype=np.int64)

    return last_text

def main():
    parser = argparse.ArgumentParser(description="Query Gemma3 model with context retrieved from a PDF vector database.")
    parser.add_argument("--query", required=True, help="The question/query to ask.")
    parser.add_argument("--db", default=None, help="Path to the vector database file.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of context chunks to retrieve.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature. Use 0.0 for greedy search.")
    args = parser.parse_args()

    # Resolve default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(args.db) if args.db else os.path.join(script_dir, "knowledge_db.pkl")
    model_path = os.path.join(script_dir, "gemma3.onnx")
    tokenizer_path = os.path.join(script_dir, "tokenizer.json")

    run_inference(
        query=args.query,
        db_path=db_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
        temperature=args.temp
    )

if __name__ == "__main__":
    main()

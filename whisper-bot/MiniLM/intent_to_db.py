#!/usr/bin/env python3
import os
import yaml
import pickle
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# --------------------------------
# Define directory and file paths
# --------------------------------
DIR = os.path.dirname(os.path.realpath(__file__))
intents_yaml_path = os.path.join(DIR, "intents.yaml")
tokenizer_path = os.path.join(DIR, "tokenizer.json")
model_path = os.path.join(DIR, "all-MiniLM.onnx")
db_path = os.path.join(DIR, "intent_db.pkl")

print(f"Directory: {DIR}")
print(f"Loading tokenizer from: {tokenizer_path}")
print(f"Loading ONNX model from: {model_path}")
print(f"Loading intents config from: {intents_yaml_path}")

# --------------------------------
# Load tokenizer and ONNX model
# --------------------------------
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
if not os.path.exists(intents_yaml_path):
    raise FileNotFoundError(f"YAML intents config file not found at: {intents_yaml_path}")

tokenizer = Tokenizer.from_file(tokenizer_path)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# --------------------------------
# Embedding generator
# --------------------------------
def get_embedding(text):
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
    
    output = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )[0]
    
    # mean pooling across sequence length (axis 0 of output[0])
    emb = np.mean(output[0], axis=0)
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb

# --------------------------------
# Parse yaml and compile database
# --------------------------------
with open(intents_yaml_path, "r") as f:
    intents_data = yaml.safe_load(f)

intent_db = {}

print("\nGenerating embeddings for intents...")
for intent, phrases in intents_data.items():
    if not phrases:
        print(f"Warning: No phrases found for intent '{intent}'. Skipping.")
        continue
    
    print(f" - {intent} ({len(phrases)} phrases)")
    vectors = []
    for phrase in phrases:
        phrase_clean = phrase.strip()
        if not phrase_clean:
            continue
        try:
            emb = get_embedding(phrase_clean)
            vectors.append(emb)
        except Exception as e:
            print(f"   Error encoding '{phrase_clean}': {e}")
            
    if vectors:
        intent_db[intent] = vectors

# --------------------------------
# Save to intent_db.pkl
# --------------------------------
print(f"\nSaving database to: {db_path}")
with open(db_path, "wb") as f:
    pickle.dump(intent_db, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nDatabase compiled successfully!")

# --------------------------------
# Verification check
# --------------------------------
print("\nVerifying intent_db.pkl...")
with open(db_path, "rb") as f:
    verify_db = pickle.load(f)

for intent, vectors in verify_db.items():
    print(f" - {intent}: {len(vectors)} vectors of shape {vectors[0].shape}")

print(f"\nVerification complete. Total intents compiled: {len(verify_db)}")

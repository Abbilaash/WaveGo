import pickle, os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# --------------------------------
# Load tokenizer
# --------------------------------

# Resolve paths relative to this file
_base_dir = os.path.abspath(os.path.dirname(__file__))

tokenizer_path = os.path.join(_base_dir, "tokenizer.json")
model_path = os.path.join(_base_dir, "all-MiniLM.onnx")

tokenizer = Tokenizer.from_file(tokenizer_path)

# --------------------------------
# Load model
# --------------------------------

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

# --------------------------------
# Load intent database
# --------------------------------

with open("intent_db.pkl", "rb") as f:
    db = pickle.load(f)

# --------------------------------
# Embedding function
# --------------------------------

def get_embedding(text):

    enc = tokenizer.encode(text)

    input_ids = np.array(
        [enc.ids],
        dtype=np.int64
    )

    attention_mask = np.ones_like(
        input_ids,
        dtype=np.int64
    )

    token_type_ids = np.zeros_like(
        input_ids,
        dtype=np.int64
    )

    output = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )[0]

    emb = np.mean(
        output[0],
        axis=0
    )

    emb = emb / np.linalg.norm(
        emb
    )

    return emb

# --------------------------------
# Intent search
# --------------------------------

def predict_intent(text):

    query_emb = get_embedding(text)

    best_intent = None
    best_score = -1

    for intent, vectors in db.items():

        for vec in vectors:

            score = np.dot(
                query_emb,
                vec
            )

            if score > best_score:

                best_score = score
                best_intent = intent

    return best_intent, best_score

# --------------------------------
# Interactive test
# --------------------------------

if __name__ == "__main__":
    while True:
        text = input("\nCommand: ")
        intent, score = predict_intent(text)
        print(f"Intent: {intent}")
        print(f"Score : {score:.4f}")
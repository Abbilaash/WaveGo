import yaml
import pickle
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ----------------------------
# Load tokenizer
# ----------------------------

tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

# ----------------------------
# Load ONNX model
# ----------------------------

session = ort.InferenceSession(
    "all-MiniLM.onnx",
    providers=["CPUExecutionProvider"]
)

# ----------------------------
# Embedding function
# ----------------------------

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

    # Mean pooling
    embedding = np.mean(
        output[0],
        axis=0
    )

    # Normalize
    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding

# ----------------------------
# Load intents
# ----------------------------

with open(
    "intents.yaml",
    "r",
    encoding="utf-8"
) as f:

    intents = yaml.safe_load(f)

# ----------------------------
# Create DB
# ----------------------------

db = {}

for intent, data in intents["intents"].items():

    print(f"Processing {intent}")

    vectors = []

    for example in data["examples"]:

        emb = get_embedding(example)

        vectors.append(emb)

        print(
            f"  {example}"
        )

    db[intent] = np.array(vectors)

# ----------------------------
# Save DB
# ----------------------------

with open(
    "intent_db.pkl",
    "wb"
) as f:

    pickle.dump(
        db,
        f
    )

print("\nSaved intent_db.pkl")
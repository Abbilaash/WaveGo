import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Resolve paths relative to this file
_base_dir = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(_base_dir, "all-MiniLM.onnx")
TOKENIZER_PATH = os.path.join(_base_dir, "tokenizer.json")

MAX_LEN = 256

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_names = [x.name for x in session.get_inputs()]

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask[..., None]
    summed = np.sum(last_hidden_state * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts

def get_embedding(text):
    encoding = tokenizer.encode(text)

    input_ids = encoding.ids[:MAX_LEN]
    attention_mask = encoding.attention_mask[:MAX_LEN]
    token_type_ids = [0] * len(input_ids)

    pad_len = MAX_LEN - len(input_ids)

    input_ids += [0] * pad_len
    attention_mask += [0] * pad_len
    token_type_ids += [0] * pad_len

    inputs = {
        input_names[0]: np.array([input_ids], dtype=np.int64),
        input_names[1]: np.array([attention_mask], dtype=np.int64),
        input_names[2]: np.array([token_type_ids], dtype=np.int64)
    }

    outputs = session.run(None, inputs)

    embedding = mean_pooling(
        outputs[0],
        np.array([attention_mask])
    )[0]

    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)

import os
import pickle
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# --------------------------------
# Load absolute paths dynamically
# --------------------------------
DIR = os.path.dirname(os.path.realpath(__file__))

tokenizer_path = os.path.join(DIR, "tokenizer.json")
model_path = os.path.join(DIR, "all-MiniLM.onnx")
db_path = os.path.join(DIR, "intent_db.pkl")

# --------------------------------
# Load tokenizer
# --------------------------------
tokenizer = Tokenizer.from_file(tokenizer_path)

# --------------------------------
# Load ONNX session
# --------------------------------
session = ort.InferenceSession(
	model_path,
	providers=["CPUExecutionProvider"]
)

# --------------------------------
# Load intent database
# --------------------------------
with open(db_path, "rb") as f:
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
	
	emb = emb / np.linalg.norm(emb)
	return emb

# --------------------------------
# Intent search
# --------------------------------
def predict_intent(text):
	query_emb = get_embedding(text)
	
	best_intent = None
	best_score = -1.0
	
	for intent, vectors in db.items():
		for vec in vectors:
			score = np.dot(query_emb, vec)
			if score > best_score:
				best_score = float(score)
				best_intent = intent
				
	return best_intent, best_score

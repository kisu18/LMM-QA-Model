from sentence_transformers import SentenceTransformer
import numpy as np 
from dataingestion import lecture_notes 
from dataingestion import llm_entries


# Load pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for lecture notes and LLM table entries
lecture_embeddings = model.encode(lecture_notes)
llm_embeddings = model.encode(llm_entries)

# Combine embeddings
embeddings = np.vstack([lecture_embeddings, llm_embeddings])


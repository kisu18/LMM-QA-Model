from dataingestion import lecture_notes,llm_entries
from embedding import model
from indexing import index


def get_relevant_texts(query, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Search the index for the top_k most similar entries
    D, I = index.search(query_embedding, top_k)

    # Retrieve the corresponding texts
    results = [lecture_notes[i] if i < len(lecture_notes) else llm_entries[i - len(lecture_notes)] for i in I[0]]
    return results

# Example query


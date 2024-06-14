import pandas as pd

# Read lecture notes from a text file
with open('source\lecture_notes.txt', 'r') as file:
    lecture_notes = file.read().split('\n')

# Read table of LLM architectures from a CSV file
llm_table = pd.read_csv('source\llm_architectures.csv')

def clean_text(text):
    # Simple text cleaning function
    text = text.replace('\n', ' ').strip()
    return text

# Clean lecture notes
lecture_notes = [clean_text(note) for note in lecture_notes]

# Convert LLM table to list of strings for embedding generation
llm_entries = llm_table.apply(lambda row: f"{row['Model']} ({row['Year']}): {row['Paper']}", axis=1).tolist()



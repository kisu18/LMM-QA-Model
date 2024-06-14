# LLM-based Question Answering System

This project implements a question-answering system using a combination of sentence embeddings, FAISS for vector indexing, and the Hugging Face Transformers library for language model responses.

## Features

- Ingests lecture notes and a table of LLM architectures.
- Generates sentence embeddings using `sentence-transformers`.
- Utilizes FAISS for efficient vector search.
- Uses the Hugging Face `distilbart-cnn-12-6` model for generating text responses.

## Setup

### Requirements

Make sure you have Python 3.7+ installed. Install the required libraries using the following command:

```bash
pip install -r requirements.txt


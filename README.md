# vasagent

This repository provides a basic Poetry environment for developing AGENTIC AI projects.

## Getting Started

1. Install [Poetry](https://python-poetry.org/docs/#installation):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install project dependencies:

   ```bash
   poetry install
   ```

3. Activate the virtual environment:

   ```bash
   poetry shell
   ```

You can now add your own AGENTIC AI code inside the `src/` directory.

## Document QA ChatBot

The `rag_app.py` module provides a basic retrieval-augmented generation (RAG) demo.
It loads a PDF, chunks the content, creates embeddings with OpenAI's
`text-embedding-3-small` model and stores them in a FAISS index. A simple
Streamlit UI lets you ask questions about the uploaded document.

Run the demo with:

```bash
streamlit run src/vasagent/rag_app.py
```


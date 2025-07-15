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

   This installs all required packages, including the `langchain-community`
   extension used for document loaders and the `tiktoken` package needed for
   OpenAI embeddings.

3. Activate the virtual environment:

   ```bash
   poetry shell
   ```

You can now add your own AGENTIC AI code inside the `src/` directory.

## Document QA ChatBot

The `rag_app.py` module provides a basic retrieval-augmented generation (RAG) demo.
It loads a PDF, chunks the content, creates embeddings with OpenAI's
`text-embedding-3-small` model and stores them in a FAISS index. A simple
Streamlit UI lets you ask questions about the uploaded document. If the PDF
contains laboratory values for **BUN** and **Creatinine**, you can press the
sidebar "Predict Kt/V" button or simply ask for a Kt/V prediction in the chat
itself. The app then calls a tool that multiplies the lab values. When the labs
are not found directly in the text, and an OpenAI API key is available, the app
falls back to a small agent that asks ChatGPT to extract the values.

Run the demo with:

```bash
streamlit run src/vasagent/rag_app.py
```


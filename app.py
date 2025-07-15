import streamlit as st
import openai
import fitz  # PyMuPDF
import faiss
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import os

# Set up your OpenAI API key
# openai.api_key = "YOUR_API_KEY"

st.title("RAG with Streamlit")

# 1. Get OpenAI API access
api_key = st.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    openai.api_key = api_key
    st.success("OpenAI API key set!")

    # 2. Load your pdf
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Chunk the PDF using PyMuPDF
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(f"PDF chunked into {len(chunks)} parts.")

        # 3. Create embeddings and store them
        @st.cache_data
        def get_embeddings(chunks):
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=chunks
            )
            return [d['embedding'] for d in response['data']]

        embeddings = get_embeddings(chunks)
        st.write("Embeddings created.")

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        st.write("FAISS index created.")

        # 4. Build a RAG pipeline
        query = st.text_input("Ask a question about the PDF:")
        if query:
            query_embedding = get_embeddings([query])[0]
            D, I = index.search(np.array([query_embedding]), k=3)

            retrieved_chunks = [chunks[i] for i in I[0]]

            prompt = f"""
            Answer the following question based on the provided context:

            Context:
            {"".join(retrieved_chunks)}

            Question: {query}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.write(response.choices[0].message['content'])

        os.remove("temp.pdf")

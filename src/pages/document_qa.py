"""Document QA chatbot page."""

import os
import tempfile
import json

import streamlit as st
from dotenv import load_dotenv

from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def run_test_prompt() -> str:
    """Run a basic OpenAI chat completion call to verify API access."""
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        return response.choices[0].message.content
    except Exception as e:  # pragma: no cover - network or auth errors
        return f"Error calling OpenAI API: {e}"


def extract_lab_values(text: str):
    """Use a chat completion to extract Urea and Creatinine values from text."""
    if not OPENAI_API_KEY:
        return None, None
    system = (
        "Extract numeric values for Urea and Creatinine from the given text. "
        "Respond only with JSON in the form {\"urea\": <value or null>, \"creatinine\": <value or null>}"
    )
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        urea = float(data["urea"]) if data.get("urea") is not None else None
        creat = float(data["creatinine"]) if data.get("creatinine") is not None else None
        return urea, creat
    except Exception:  # pragma: no cover - network or auth errors
        return None, None


def show() -> None:
    """Display the Document QA chatbot page."""
    st.title("Document QA ChatBot")

    st.sidebar.write(
        "Upload a PDF and ask questions. Embeddings are created with"
        " OpenAI 'text-embedding-3-small' and stored in a local FAISS index."
    )

    if st.sidebar.button("Test OpenAI API"):
        st.sidebar.write(run_test_prompt())

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_content = " ".join(doc.page_content for doc in docs)
        urea_val, creat_val = extract_lab_values(text_content)
        enable_ktv = urea_val is not None and creat_val is not None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        vectors = FAISS.from_documents(chunks, embeddings)

        prompt = ChatPromptTemplate.from_template(
            """
Answer the questions based on the provided text only. If the answer is not
contained in the text, say so.

<context>
{context}
</context>
Question: {input}
"""
        )

        document_chain = create_stuff_documents_chain(ChatOpenAI(api_key=OPENAI_API_KEY), prompt)
        retriever = vectors.as_retriever()
        qa_chain = create_retrieval_chain(retriever, document_chain)

        if enable_ktv:
            if st.sidebar.button("Predict Kt/V"):
                st.sidebar.write(f"Kt/V: {urea_val * creat_val}")
        else:
            st.sidebar.button("Predict Kt/V", disabled=True)

        user_query = st.chat_input("Ask a question:")
        if user_query:
            result = qa_chain.invoke({"input": user_query})
            st.write(result.get("answer"))
    else:
        st.write("Please upload a PDF to begin.")

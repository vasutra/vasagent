# Streamlit RAG application
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

try:
    import openai
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.prompts import ChatPromptTemplate
    from langchain.chat_models import ChatOpenAI
except Exception as e:  # pragma: no cover - optional dependencies
    openai = None
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    OpenAIEmbeddings = None
    FAISS = None
    create_retrieval_chain = None
    create_stuff_documents_chain = None
    ChatPromptTemplate = None
    ChatOpenAI = None


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def run_test_prompt():
    """Run a basic OpenAI chat completion call to verify API access."""
    if openai is None:
        return "openai package not available"

    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            api_key=OPENAI_API_KEY,
        )
        return response.choices[0].message["content"]
    except Exception as e:  # pragma: no cover - network or auth errors
        return f"Error calling OpenAI API: {e}"


st.set_page_config(
    page_title="Document QA ChatBot",
    page_icon=":robot_face:",
    layout="centered",
)

st.title("Document QA ChatBot")

st.sidebar.write(
    "Upload a PDF and ask questions. Embeddings are created with"
    " OpenAI 'text-embedding-3-small' and stored in a local FAISS index."
)

if st.sidebar.button("Test OpenAI API"):
    st.sidebar.write(run_test_prompt())

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and PyPDFLoader is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

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

    user_query = st.chat_input("Ask a question:")
    if user_query:
        result = qa_chain.invoke({"input": user_query})
        st.write(result.get("answer"))
else:
    st.write("Please upload a PDF to begin.")

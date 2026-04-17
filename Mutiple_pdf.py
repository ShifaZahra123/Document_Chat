import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ---------------- API KEY ----------------
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------- PDF TEXT EXTRACTION ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# ---------------- CHUNKING ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


# ---------------- VECTOR STORE ----------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)


# ---------------- GEMINI ANSWER (SAFE - NO LANGCHAIN LLM) ----------------
def ask_gemini(context, question):

    prompt = f"""
    You are a helpful assistant.

    Answer ONLY using the context below.

    If answer is not found, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


# ---------------- USER QUERY ----------------
def user_input(user_question, vector_store):

    docs = vector_store.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    answer = ask_gemini(context, user_question)

    st.write("Reply:", answer)


# ---------------- STREAMLIT APP ----------------
def main():

    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("📄 Chat with Your PDFs")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.title("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):

                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(chunks)

                    st.success("Done! You can now ask questions.")
            else:
                st.warning("Please upload PDF files first.")


if __name__ == "__main__":
    main()

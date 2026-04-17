import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------- PDF TEXT ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# ---------------- CHUNKS ----------------
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


# ---------------- QA CHAIN ----------------
def get_conversational_chain():

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context.

    If the answer is not in the context, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    return create_stuff_documents_chain(model, prompt)


# ---------------- USER INPUT ----------------
def user_input(user_question, vector_store):

    retriever = vector_store.as_retriever()

    document_chain = get_conversational_chain()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": user_question
    })

    st.write("Reply:", response["answer"])


# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF 💁")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a question from the PDF")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(chunks)
                st.success("Done")


if __name__ == "__main__":
    main()

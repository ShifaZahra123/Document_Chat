import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Convert text into Chunks
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Provide Embeddings by Google by using Google API Keys, Vector Embedding Technique (Convert Chunks of Text to Vectors)
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Vector store DB created by Facebook doing Similarity Search
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat with documnets
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):  # Text Extraction From pdf
    text=""
    for pdf in pdf_docs:  
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:  # Pdf reader is in form of list
            text+= page.extract_text()
    return  text


def get_text_chunks(text):   # Text Convert into Smaller chunks of size 10000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) # To avid missing texts from pdf and chunk_size is large so chunk_overlap also becomes larger
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"]) # langchain function PromptTemplate()
    def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {input}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = create_stuff_documents_chain(model, prompt)

    return chain

    return chain



from langchain.chains import create_retrieval_chain

def user_input(user_question, vector_store):

    retriever = vector_store.as_retriever()

    document_chain = get_conversational_chain()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": user_question
    })

    st.write("Reply:", response["answer"])


def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF💁")
    
    if 'vector_store' not in st.session_state:  # Keep vector_store in session state
        st.session_state['vector_store'] = None

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and st.session_state['vector_store']:
        user_input(user_question, st.session_state['vector_store'])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state['vector_store'] = get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

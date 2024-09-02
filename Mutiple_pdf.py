import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Convert text into Chunks
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Provide Embeddings by Google by using Google API Keys, Vector Embedding Technique (Convert Chunks of Text to Vectors)
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Vector store DB created by Facebook doing Similarity Search
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat with documnets
from langchain.chains.question_answering import load_qa_chain # For chat with documnets
from langchain.prompts import PromptTemplate
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
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # Storing Database in Local Environment


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
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # For internal text summarization: chain_type="stuff"

    return chain



def user_input(user_question): # text box functionality
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

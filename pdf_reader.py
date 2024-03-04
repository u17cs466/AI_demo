'''
create .env file at project level and create OPENAI_API_KEY also PINECONE_API_KEY
example:- OPENAI_API_KEY="YOUR API KEY"
          PINECONE_API_KEY="YOUR API KEY"

 create requirements.txt
       pinecone
       langchain
       langchain_pinecone
       dotenv
  
'''
from pinecone import Pinecone
import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()

import os
openai_key=os.getenv("OPENAI_API_KEY")
pinecone_key=os.getenv("PINECONE_API_KEY")

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    chunks = text_splitter.split_text(doc)
    return chunks


def get_vector_store(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    index_name = "langchainvector"    
    docsearch = PineconeVectorStore(pinecone_api_key=pinecone_key,embedding=embeddings,index_name=index_name)
    docsearch.add_texts(texts=documents)
    return  docsearch


def get_conversational_chain():
    llm = OpenAI(model="text-davinci-003",temperature=0.5,openai_api_key=openai_key)
    chain=load_qa_chain(llm=llm,chain_type="stuff")
    return chain

def query_search(query,k=2):
    documents=get_vector_store()
    matching_results= documents.similarity_search(query,k=k)
    return matching_results


def user_input(user_question):
    get_conversational_chain()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    index_name = "langchainvector"    
    docsearch = PineconeVectorStore(pinecone_api_key=pinecone_key,embedding=embeddings,index_name=index_name)
    matching_results= docsearch.similarity_search(user_question,k=2)
    st.write(matching_results)
    
    
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI💁")

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
                st.success("You Have Uploaded Your PDF to vectordb \n Please Ask your Query")



if __name__ == "__main__":
    main()

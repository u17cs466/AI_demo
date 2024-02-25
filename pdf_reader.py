'''
create .env file at project level and create OPENAI_API_KEY also PINECONE_API_KEY
example:- OPENAI_API_KEY="YOUR API KEY"
          PINECONE_API_KEY="YOUR API KEY"
'''
from pinecone import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
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

doc=read_doc('pdf_dir/')


def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document=text_splitter.split_documents(docs)
    return document


documents=chunk_data(docs=doc)


embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
# vector = embeddings.embed_query("good morning")
# len(vector)


pc = Pinecone(
        api_key=pinecone_key
    )

if 'langchainvector' not in pc.list_indexes().names():
    pass
pc._get_status("langchainvector")


index_name = "langchainvector"
pinecone_api_key=pinecone_key

docsearch = PineconeVectorStore(pinecone_api_key=pinecone_key,embedding=embeddings,index_name=index_name)
docsearch.add_documents(documents=documents)


llm = OpenAI(model="text-davinci-003",temperature=0.5,openai_api_key=openai_key)
chain=load_qa_chain(llm=llm,chain_type="stuff")

def query_search(query,k=2):
    matching_results= docsearch.similarity_search(query,k=k)
    return matching_results

query = "how the agriculture is doing?"
search_output = query_search(query)
print(search_output)

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon="logo1.png" )
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        

#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)
                


#                 # Clear chat history
#                 st.session_state.chat_history = None
                
#     if st.session_state.conversation is not None:
#         if st.session_state.chat_history is None:
#             # Greet the user
#             greeting = "Hello! How can I assist you with your documents?"
#             st.write(bot_template.replace("{{MSG}}", greeting), unsafe_allow_html=True)

# if __name__ == '__main__':
#     main()

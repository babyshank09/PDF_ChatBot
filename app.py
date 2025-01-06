import os 
import dotenv  
from dotenv import load_dotenv   
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain  
from langchain.chains import create_retrieval_chain 




st.set_page_config(page_title="PDF Chatbot") 
st.title("PDF ChatBot") 
st.subheader("Interpret your PDF Files and Chat with it") 


def create_vector_embedding(openai_api_key): 
    st.session_state.loader= PyPDFDirectoryLoader("pdf_files")  
    st.session_state.docs= st.session_state.loader.load()
    st.session_state.splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)  
    st.session_state.final_docs= st.session_state.splitter.split_documents(st.session_state.docs)  
    st.session_state.embedder= OpenAIEmbeddings(openai_api_key= openai_api_key)
    st.session_state.vectorstore= FAISS.from_documents(st.session_state.final_docs, embedding=st.session_state.embedder) 
    st.session_state.retriever= st.session_state.vectorstore.as_retriever() 
    


with st.sidebar:  
    st.title("Settings")
    groq_api_key= st.text_input("Enter your Groq API Key:",type="password")    
    openai_api_key= st.text_input("Enter your OPENAI API Key:", type="password")
    uploaded_files= st.file_uploader("Upload your files",type=["pdf"], accept_multiple_files=True)   
    st.write(len(uploaded_files)) 
    
    os.makedirs("pdf_files", exist_ok=True)  
    i=1
    if uploaded_files:
        for doc in uploaded_files:
            file_name= "pdf_files/PDF_" + str(i) + ".pdf"
            with open(file_name, 'wb') as file: 
                file.write(doc.getvalue()) 
            i+=1  
        st.success("Files uploaded successfully") 


    if not st.button("Embed Document"): 
        st.info("Make sure to embed your documents before proceeding") 
    else:  
        with st.spinner():
            create_vector_embedding(openai_api_key) 
            st.success("Vector Embeddings created successfully")  

    
if groq_api_key:
    llm= ChatGroq(groq_api_key= groq_api_key, model="Llama3-8b-8192")    

prompt= ChatPromptTemplate.from_template(
    """ 
    "You are a helpful assistant. Answer the questions based on the given context only 
    <context> 
    "{context}" 
    </context> 
    Question: "{input}" 
    """
)

query= st.text_area("Ask me anything") 

if st.button("Submit") and query:  
    with st.spinner():
        document_chain= create_stuff_documents_chain(llm, prompt) 
        retrieval_chain= create_retrieval_chain(st.session_state.retriever, document_chain)  
        
        response= retrieval_chain.invoke({"input":query})  
        st.write(response['answer']) 














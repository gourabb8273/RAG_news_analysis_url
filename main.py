import os
import streamlit as st
import pickle
import time
import langchain

from dotenv import  load_dotenv

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

from huggingface_hub import login

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

load_dotenv()
query =False
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


st.title("Welcome News Research Tool 📈")

st.sidebar.title("News Article URLs")

# st.sidebar.header("How many urls you want to submit?")
num_sources = st.sidebar.number_input("Enter no of sources urls", min_value=1, max_value=10, value=1, step=1)

import nltk
nltk.download('punkt')

main_placeholder = st.empty()

# print(num_sources)
urls=[]
data =''
file_path = "./faiss_vector_store.pickle"

for i in range(num_sources):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url = st.sidebar.button(f"Process URLs") # return true if clicked the button


# Load the data from urls
if process_url:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    main_placeholder.text(" Data is loading .....🚀")
    # Create a chunk out from data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200 # to provide some context
    )
    main_placeholder.text("Text splitting started .....🚀")
    docs = text_splitter.split_documents(data)

    # create vector embedding
    huggingface_token = HUGGING_FACE_TOKEN
    login(token=huggingface_token)
    main_placeholder.text("Embedding vector has started .....🚀")
    # Step 2: Initialize HuggingFaceEmbeddings with a specific model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Store in Faiss vector database
    main_placeholder.text("Storing in Vector db .....🚀")
    vector_store = FAISS.from_documents(docs,embeddings)
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
            pickle.dump(vector_store, f)



query = main_placeholder.text_input("Enter your Question")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_retries=2
            )

# retriever = vector_store.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
# )


if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
                vector_store = pickle.load(f)
                qa_chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, retriever=vector_store.as_retriever()
                    ) 
                result = qa_chain({"question": query})
                main_placeholder.text("Finding your answer .....🚀")
                st.subheader("Answer")
                st.subheader(result['answer'])
                sources = result.get("sources","")
                if sources:
                        st.subheader("Sources: ")
                        sources_list = sources.split("\n") #split the source by newline
                        for source in sources_list:
                                st.write(source)
              




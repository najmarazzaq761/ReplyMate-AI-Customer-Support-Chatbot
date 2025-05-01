# importing libraries
import streamlit as st
import pandas as pd
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# page configuration
st.title("ReplyMate AI")
with st.sidebar:
    st.image("hack4.jpg", use_container_width=True)
    st.markdown(
        "**ReplyMate AI** is your smart e-commerce assistant that instantly answers customer queries about products. "
        "Enhance your customer support with fast, accurate, and AI-powered responses from your live product database."
    )
    st.title("Configuration")
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# fetch Products Data form database
def fetch_product_data():
    headers = {
        "apikey": st.secrets["SUPABASE_API_KEY"],
        "Authorization": f"Bearer {st.secrets['SUPABASE_API_KEY']}",
    }
    url = "https://uqrpmglchvatwkzonyva.supabase.co/rest/v1/products"
    response = requests.get(url, headers=headers)
    data = response.json()
    return "\n".join([
        f"Product: {item['name']}\nDescription: {item['description']}\nPrice: ${item['price']}\n"
        for item in data
    ])

# loading data
@st.cache_data
def load_data():
    return fetch_product_data()

# creating data chunks
@st.cache_data
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.create_documents([_data])


# creating vectorstores
@st.cache_resource
def create_vectorstores(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process
data = load_data()
docs = split_data(data)
vectorstores = create_vectorstores(docs)

# LLM and retriever
retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=temp,
    max_tokens=None,
    timeout=None
)

# prompt template
system_prompt = (
    "You are an assistant for product question-answering tasks. "
    "Use the following product database context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# session state for chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# chat input
query = st.chat_input("Ask a product-related question")
if query:
    st.chat_message("user").write(query)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    # display and store
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# importing libraries
import streamlit as st
import pandas as pd
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# page configuration
st.title("🤖ReplyMate AI")
with st.sidebar:
    st.image("hack4.jpg", use_container_width=True)
    st.markdown(
        "**ReplyMate AI** is your smart e-commerce assistant that instantly answers customer queries about products. "
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    return text_splitter.create_documents([_data])


# creating vectorstores
@st.cache_resource
def create_vectorstores(_docs):
    embeddings = embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process
data = load_data()
docs = split_data(data)
vectorstores = create_vectorstores(docs)

# LLM and retriever
retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=temp,
    max_tokens=None,
    timeout=None
)

# prompt template
system_prompt = (
    "You are an assistant for product question-answering tasks for an E-commerce Company named as NR Colloections. "
    "Use the following product database context to answer the question. "
    "Please always give answer in proper format."
    "If you don't know the answer, say you don't know. "
    "\n\n"
    "If the user sends a greeting (like 'hi', 'hello', 'hey' , or 'what you can do'), respond with a friendly greeting, "
    "introduce yourself as a ReplyMate AI, and let them know you're available to assist with any questions about NR Collecitions. "
    "Also, ask: 'How can I help you today?'"
    "please only greet them once and then only give answer to queries and don't introduce yourself with every answer\n\n"
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

# clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask Question"}]
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

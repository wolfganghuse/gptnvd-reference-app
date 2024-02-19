# Import necessary libraries
import logging
import streamlit as st
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from KserveML import KserveML
#from langchain.memory.chat_message_histories import RedisChatMessageHistory
#from langchain.memory import ConversationBufferMemory
from torch import cuda
from langchain.vectorstores import Milvus
import uuid

from prometheus_client import Counter, Histogram, start_http_server

from config import *

def generate_session_id():
    return str(uuid.uuid4())

@st.cache_resource()
def load_llm():
    #"http://llama2chat.llm1.kubeflow4.gptnvd.dachlab.net/v2/models/llama2chat_7b/infer"
    llm = KserveML(
      endpoint_url=INFERENCE_ENDPOINT
    )

    return llm

@st.cache_resource()
def connect_redis():
    message_history = RedisChatMessageHistory(
        url='redis://:'+REDIS_PASSWORD+'@'+REDIS_URL, ttl=180, session_id=generate_session_id()
    )

    return message_history


@st.cache_resource()
def load_vector_store():
    #Injected in sentence_base container image
    modelPath="sentence-transformers/all-mpnet-base-v2"

    device = f'cuda' if cuda.is_available() else 'cpu'

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,  
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )


    # openthe vector store database
    vector_db = Milvus(
        embeddings,
        collection_name = MILVUS_COLLECTION,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    return vector_db


# Set the title for the Streamlit app
st.title("Simple RAG Pipeline on GPT for Nutanix - 🦜🦙")

# Create a file uploader in the sidebar
#uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")
st.sidebar.write("""
Reference Application on GPT-in-a-Box for Nutanix (0.1)
=========================================
                 
Infrastructure
--------------
- Nutanix NX with GPU
- AOS 6.7
- PC2023.4
- Files
- Objects

Kubernetes Infrastructure
-------------------------
- NKE 2.9
- Kubernetes 1.25.6
- MetalLB
- GPU-enabled Worker Node Pool
- NAI-LLM-k8s with Helm Chart
- Jupyter Lab for Experiments
- Milvus Vectorstore
- Redis Cache

Application Architecture
========================

Overview:
---------
- Llama2 based RAG Pipeline with Conversational Memory and RAG-Pipeline for domain-specific knowledge


Details:
--------
- Llama2 7B Large Language Model running on kserve inference service
- RAG-pipeline using Milvus Vectorstore 
- Conversational memory uses Redis Cache
- Streamlit-based Frontend
""")

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
# Handle file upload
# Load the language model
llm = load_llm()
db = load_vector_store()
message_history = connect_redis()

memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)
memory.clear()
# Create a conversational chain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=db.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    memory=memory
)

# Function for conversational chat
def conversational_chat(query):
    try:
        result = chain({"question": query})
        logging.info(f"Query processed: {query}")
        return result["answer"]
    except Exception as e:
        logging.error(f"Error processing query: {query}, Error: {e}")
        return "Sorry, I couldn't process your request."
    
# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input('I can work on Data in your Document Bucket... ask me anything', "", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        logging.info(f"User input received: {user_input}")
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
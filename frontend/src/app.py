# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
from langchain import hub
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA
from torch import cuda
from config import *


ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


# Set up RetrievelQA model
rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")

device = f'cuda' if cuda.is_available() else 'cpu'

#modelPath = "sentence-transformers/all-mpnet-base-v2"
modelPath = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

vectorstore = Milvus(
    embeddings,
    collection_name = MILVUS_COLLECTION,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)

def load_model():
    #"http://llama2chat.llm1.kubeflow4.gptnvd.dachlab.net/v2/models/llama2chat_7b/infer"
    llm = KserveML(
      endpoint_url=INFERENCE_ENDPOINT
    )

    return llm


def retrieval_qa_chain(llm, vectorstore):

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True,
    )
    return qa_chain


def qa_bot():
    llm = load_model()

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to GPT Reference App build with Chainlit and LangChain."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

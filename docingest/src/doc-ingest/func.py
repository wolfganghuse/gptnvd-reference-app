from parliament import Context
from cloudevents.conversion import to_json
from config import *

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from prometheus_client import Counter, Histogram, start_http_server

import logging
from torch import cuda

import boto3




REQUESTS = Counter(
    'doc_ingest_requests_total', 
    'Total number of requests' 
)
TOTAL_LATENCY = Histogram(
    'doc_ingest_total_latency_seconds', 
    'Latency of ingestion in seconds'
)

EMBED_LATENCY = Histogram(
    'doc_ingest_embed_latency_seconds', 
    'Latency of embedding in seconds'
)

S3_LATENCY = Histogram(
    'doc_ingest_s3_latency_seconds', 
    'Latency of s3 retrieval in seconds'
)

# Start a Prometheus metrics server on port 8001
start_http_server(8001)

FORMAT = f'%(asctime)s %(id)-36s {FUNC_NAME} %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('boto3')
logger.setLevel(logging.DEBUG)

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

def s3_client():

    session = boto3.Session()
    return session.client('s3',
                          endpoint_url=S3_ENDPOINT_URL,
                          aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_KEY,
                          region_name=S3_REGION,
                          verify=SSL_VERIFY)


def get_signed_url(bucket, obj):
    # The number of seconds the presigned url is valid for. By default it expires in an hour (3600 seconds)
    SIGNED_URL_EXPIRATION = os.environ.get('SIGNED_URL_EXPIRATION', 3600)

    return s3_client().generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': obj},
        ExpiresIn=SIGNED_URL_EXPIRATION
    )


def main(context: Context):

    with TOTAL_LATENCY.time():

        REQUESTS.inc()

        source_attributes = context.cloud_event.get_attributes()

        logger.info(
            f'REQUEST:: {to_json(context.cloud_event)}', extra=source_attributes)

        data = context.cloud_event.data
        notificationType = data["Records"][0]["eventName"]
        if notificationType == "s3:ObjectCreated:Put" or notificationType == "s3:ObjectCreated:CompleteMultipartUpload":
            srcBucket = data["Records"][0]["s3"]["bucket"]["name"]
            srcObj = data["Records"][0]["s3"]["object"]["key"]
            with S3_LATENCY.time():
                signed_url = get_signed_url(
                    srcBucket, srcObj)
                logger.info(f'SIGNED URL:: {signed_url}', extra=source_attributes)

                try:
                    loader = PyPDFLoader(signed_url)
                except:
                    logger.info(f'URL not found:: {signed_url}', extra=source_attributes)
                    return "", 500
                try:
                    docs = loader.load()
                except:
                    logger.info(f'Error while processing:: {signed_url}', extra=source_attributes)
                    return "", 500
                
            # Split loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
            chunked_documents = text_splitter.split_documents(docs)

            MILVUS_HOST = os.environ['MILVUS_HOST']

            with EMBED_LATENCY.time():
                vector_db = Milvus.from_documents(
                    chunked_documents,
                    embeddings,
                    collection_name = MILVUS_COLLECTION,
                    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                )
                logger.info(f'Successfully embedded:: {notificationType}', extra=source_attributes)

        else:
            logger.info(f'Not processing:: {notificationType}', extra=source_attributes)


        return "", 204

from parliament import Context
from cloudevents.conversion import to_json
from config import *

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

from prometheus_client import Counter, Histogram, start_http_server

import logging
from datetime import datetime

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

MILVUS_LATENCY = Histogram(
    'doc_ingest_milvus_request_latency_seconds', 
    'Latency of Milvus requests in seconds'
)

FORMAT = f'%(asctime)s %(id)-36s {FUNC_NAME} %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('boto3')
logger.setLevel(logging.INFO)


def s3_client():

    session = boto3.Session()
    return session.client('s3',
                          endpoint_url=AWS_S3_ENDPOINT_URL,
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                          region_name=AWS_REGION,
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

    source_attributes = context.cloud_event.get_attributes()

    logger.info(
        f'REQUEST:: {to_json(context.cloud_event)}', extra=source_attributes)

    data = context.cloud_event.data
    notificationType = data["Records"][0]["eventName"]
    if notificationType == "s3:ObjectCreated:Put":
        srcBucket = data["Records"][0]["s3"]["bucket"]["name"]
        srcObj = data["Records"][0]["s3"]["object"]["key"]
        signed_url = get_signed_url(
            srcBucket, srcObj)
        logger.info(f'SIGNED URL:: {signed_url}', extra=source_attributes)

        modelPath = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            cache_folder='/app/model',
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs 
        )

        loader = WebBaseLoader(signed_url)

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        splitted_docs = text_splitter.split_documents(docs)

        MILVUS_HOST = os.environ['MILVUS_HOST']

        vector_db = Milvus.from_documents(
            splitted_docs,
            embeddings,
            collection_name = 'doc_ingest',
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        )
        logger.info(f'Successfully embedded:: {notificationType}', extra=source_attributes)

    else:
        logger.info(f'Not processing:: {notificationType}', extra=source_attributes)


    return "", 204

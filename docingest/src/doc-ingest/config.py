import os

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
MILVUS_COLLECTION = os.environ.get('MILVUS_COLLECTION')


FUNC_NAME = os.environ.get('K_SERVICE', 'doc-ingest')

SSL_VERIFY = os.environ.get("SSL_VERIFY")
S3_REGION = os.environ.get('S3_REGION')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')
S3_ENDPOINT_URL = os.environ.get('S3_ENDPOINT_URL')

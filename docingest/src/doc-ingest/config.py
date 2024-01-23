import os

MILVUS_HOST = os.environ.get('MILVUS_HOST','milvus.gpt01mgmt01.cloudnative.nvdlab.net')
MILVUS_PORT = os.environ.get('MILVUS_PORT','19530')
MILVUS_COLLECTION = os.environ.get('MILVUS_COLLECTION','collection01')

SSL_VERIFY = os.environ.get("SSL_VERIFY", "False")
FUNC_NAME = os.environ.get('K_SERVICE', 'doc-ingest')

AWS_REGION = os.environ.get('AWS_REGION','us-east-1')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID','aEFfT0qYGqejlVsrqWTYH8p2uOhpa9YT')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY','tEQjKskC_RWlFoHEkzWGMfIlJ1-oAxOY')
AWS_S3_ENDPOINT_URL = os.environ.get('AWS_S3_ENDPOINT_URL','objects.gptnvd.cloudnative.nvdlab.net')

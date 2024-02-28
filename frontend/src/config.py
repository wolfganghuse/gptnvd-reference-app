import os
MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
MILVUS_COLLECTION = os.environ.get('MILVUS_COLLECTION')

INFERENCE_ENDPOINT = os.environ.get('INFERENCE_ENDPOINT')

SSL_VERIFY = os.environ.get("SSL_VERIFY")
S3_REGION = os.environ.get('S3_REGION')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')
S3_ENDPOINT_URL = os.environ.get('S3_ENDPOINT_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

# REDIS_URL = os.environ.get('REDIS_URL','redis-master.redis:6379')
# REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD','xtOygnkYIS')
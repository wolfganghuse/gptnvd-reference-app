import os
MILVUS_HOST = os.environ.get('MILVUS_HOST','milvus.gpt01mgmt01.cloudnative.nvdlab.net')
MILVUS_PORT = os.environ.get('MILVUS_PORT','19530')
MILVUS_COLLECTION = os.environ.get('MILVUS_COLLECTION','collection02')

INFERENCE_ENDPOINT = os.environ.get('INFERENCE_ENDPOINT','http://llm.llm.gpt01dev01.cloudnative.nvdlab.net/v2/models/llama2_7b_chat/infer')

# REDIS_URL = os.environ.get('REDIS_URL','redis-master.redis:6379')
# REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD','xtOygnkYIS')
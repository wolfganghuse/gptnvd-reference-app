from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import *
from utilities import (
    provision_embedding_model,
    connect_milvus,
    connect_elastic_search,
)

from prometheus_client import Counter, Histogram, start_http_server
import time


REQUESTS = Counter(
    'doc_retrieval_requests_total', 
    'Total number of requests' 
)
REQUEST_LATENCY = Histogram(
    'doc_retrieval_request_latency_seconds', 
    'Latency of requests in seconds'
)
MILVUS_LATENCY = Histogram(
    'doc_retrieval_milvus_request_latency_seconds', 
    'Latency of Milvus requests in seconds'
)
ELASTIC_LATENCY = Histogram(
    'doc_retrieval_elastic_request_latency_seconds', 
    'Latency of Elasticsearch requests in seconds'
)

app = FastAPI()

# Start a Prometheus metrics server on port 8001
start_http_server(8001)

# embedding model
embdeding_model = provision_embedding_model()

# Milvus
vector_store = connect_milvus(embdeding_model)

# elastic search
es_client = connect_elastic_search()


# query format
class Query(BaseModel):
    query: str
    num_docs: Optional[int] = 2
    title_weight: Optional[float] = 1.0

@app.get("/health")
def read_health():
    return {"OK"}

@app.post("/query/")
async def retrieve_documents(query: Query):
    with REQUEST_LATENCY.time():
        start_time = time.time()

        REQUESTS.inc()

        # validate input
        query_str = query.query.strip()
        if not query_str:
            raise HTTPException(
                status_code=400,
                detail="Empty query!"
            )

        if query.num_docs > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximun number of documents to retrieve is 10!"
            )
        num_docs = query.num_docs

        if query.title_weight < 0:
            raise HTTPException(
                status_code=400,
                detail="title_weight must be non-negative!"
            )
        title_weight = query.title_weight

        # vector search
        with MILVUS_LATENCY.time():
            vector_docs = vector_store.similarity_search_with_score(
                query_str,
                k=10
            )
        # apply title weight and rerank
        vector_res = []
        for doc, doc_score in vector_docs:
            if doc.metadata['part'] == "title":
                vector_res.append([doc.metadata['id'], doc_score * title_weight])
            else:
                vector_res.append([doc.metadata['id'], doc_score])
        vector_res.sort(key=lambda x:x[1], reverse=True)

        return_docs = []
        return_scores = []

        print (vector_res[0][1])
        print (QUALITY_THRESHOLD)
        # if vector retrieval result is not good, turn to elastic search
        if vector_res[0][1] < QUALITY_THRESHOLD:
            # try non-fuzzy first
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query_str,
                        "fields": [f"theme^{title_weight}", "content"],
                        "fuzziness": 0
                    }
                },
                "size": num_docs
            }
            with ELASTIC_LATENCY.time():
                es_docs = es_client.search(index=ELASTIC_INDEX, body=es_query)

            # then try fuzzy
            if len(es_docs['hits']['hits']) == 0:
                es_query["query"]["multi_match"]["fuzziness"] = "AUTO"
                es_docs = es_client.search(index=ELASTIC_INDEX, body=es_query)
            
            if len(es_docs['hits']['hits']) > 0:
                return_docs = [
                    hit['_source'] for hit in es_docs['hits']['hits'][:num_docs]
                ]
                return_scores = [
                    hit['_score'] for hit in es_docs['hits']['hits'][:num_docs]
                ]

        # elif vector retrieval result is good, or elastic search also failed
        if len(return_docs) == 0:
            return_ids = set()
            for doc_id, doc_score in vector_res:
                if doc_score >= RELAVANCE_THRESHOLD and doc_id not in return_ids:
                    return_docs.append(
                        es_client.get(
                            index=ELASTIC_INDEX,
                            id=doc_id,
                        )['_source']
                    )
                    return_scores.append(doc_score)
                    return_ids.add(doc_id)
                    if len(return_scores) == num_docs:
                        break
        execution_time = time.time() - start_time
        return {
            'documents': return_docs,
            'scores': return_scores,
            'execution_time': execution_time  # Optionally return the execution time
        }

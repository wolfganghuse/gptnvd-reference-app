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

Application Architecture
========================

Overview:
---------
- Llama2 based RAG Pipeline for domain-specific knowledge


Details:
--------
- Llama2 7B Large Language Model running on kserve inference service
- BGE-large-en-v1.5 embedding model
- RAG-pipeline using Milvus Vectorstore 
- Chainlit-based Frontend

<img src="ref-app.png" alt="Quick Start"></img>
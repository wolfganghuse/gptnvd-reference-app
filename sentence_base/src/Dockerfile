FROM python:3.10

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

USER 1001

WORKDIR $APP_HOME

COPY requirements.txt ./

USER 0

ENV HF_HOME /app/model 

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder='/app/model')"

RUN python -c "import transformers; transformers.utils.move_cache()"
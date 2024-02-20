FROM quay.io/wolfgangntnx/sentence_base:latest

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY requirements.txt ./

USER 0

RUN pip install --no-cache-dir -r requirements.txt

USER 1001

COPY . ./

ENV TRANSFORMERS_CACHE /app/model 
ENV SENTENCE_TRANSFORMERS_HOME /app/model

EXPOSE 8000
 
ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=8000", "--headless"]
FROM quay.io/wolfgangntnx/sentence_base:latest

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

USER 0

RUN pip install --no-cache-dir -r requirements.txt

USER 1001

ENV TRANSFORMERS_CACHE /app/model 
ENV SENTENCE_TRANSFORMERS_HOME /app/model

EXPOSE 8501
 
CMD python -m parliament .

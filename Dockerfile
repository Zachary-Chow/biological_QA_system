FROM python:3.10

WORKDIR /app

COPY . /app

# 复制模型文件到容器中的 /app/models 路径下
COPY /model/models--BAAI--bge-m3/snapshots/test /app/model

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade --retries 5 --default-timeout=100000 -r requirements.txt



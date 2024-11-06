FROM python:3.10

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade --retries 5 --default-timeout=100000 -r requirements.txt



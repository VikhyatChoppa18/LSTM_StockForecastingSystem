FROM  python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .


CMD ["streamlit","run","main.py"]



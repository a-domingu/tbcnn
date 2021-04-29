# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster


WORKDIR /app

ADD sets /app/sets
ADD logs /app/logs

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["python3", "application.py"]

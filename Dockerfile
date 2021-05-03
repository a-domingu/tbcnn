# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

ADD sets /app/sets
ADD test /app/test
ADD logs /app/logs
# Add sample application
ADD main.py /app/main.py

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8000

# Run it
CMD ["python3", "/app/main.py"]

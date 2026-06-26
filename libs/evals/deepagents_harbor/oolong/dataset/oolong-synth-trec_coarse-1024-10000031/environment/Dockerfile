FROM python:3.13-slim

RUN pip install --no-cache-dir python-dateutil==2.9.0.post0

RUN mkdir -p /app
WORKDIR /app

COPY context.txt /app/context.txt

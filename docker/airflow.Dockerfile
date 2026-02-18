FROM apache/airflow:2.10.2-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow


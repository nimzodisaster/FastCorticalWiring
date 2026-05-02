FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libsuitesparse-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install -vvv --no-cache-dir --no-binary potpourri3d potpourri3d

COPY . .

ENTRYPOINT ["/bin/bash", "-c", "python backend_check.py && python fastcw.py --help"]

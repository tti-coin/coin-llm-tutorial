FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonライブラリのインストール
COPY requirements.txt /tmp
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

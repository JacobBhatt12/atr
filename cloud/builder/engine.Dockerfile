# VRTX Engine image for model deployments (built by cloud/builder/builder.py).
# CPU torch by default; for NVIDIA workers set VRTX_TORCH_INDEX=https://download.pytorch.org/whl/cu121
FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
WORKDIR /vrtx
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY engine engine
COPY gateway gateway
ENV PORT=8000
EXPOSE 8000
CMD ["/bin/sh", "-c", "exec python -m uvicorn gateway.server:app --host 0.0.0.0 --port $PORT"]

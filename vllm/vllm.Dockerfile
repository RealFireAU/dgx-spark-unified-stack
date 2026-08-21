FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y python3-pip python3 curl git libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages vllm --extra-index-url https://wheels.vllm.ai/nightly/

WORKDIR /app

ENV VLLM_FLASHINFER_MOE_BACKEND=latency
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

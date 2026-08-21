FROM vllm-spark:local

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI
RUN sed -i -E '/^torch(vision|audio)?([<>=~! ].*)?$/d' requirements.txt && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

# ComfyUI-Manager and ComfyUI-Model-Manager dependencies
RUN pip install --no-cache-dir --break-system-packages \
    GitPython PyGithub matrix-nio transformers huggingface-hub \
    typer rich typing-extensions toml uv chardet markdownify

EXPOSE 8188
ENTRYPOINT ["python3", "main.py"]
CMD ["--listen", "0.0.0.0", "--port", "8188"]

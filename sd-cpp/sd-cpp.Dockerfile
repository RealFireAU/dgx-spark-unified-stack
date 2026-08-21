FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/cuda/targets/sbsa-linux/lib/stubs && ln -sf libcuda.so libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib/stubs:$LIBRARY_PATH

WORKDIR /app
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp src

RUN cd src && mkdir build && cd build && \
    cmake .. \
    -DSD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link=/usr/local/cuda/targets/sbsa-linux/lib/stubs" \
    && make -j$(nproc) sd-server

FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/src/build/bin/sd-server /app/sd-server
COPY --from=builder /app/src/build/bin/*.so* /usr/lib/

EXPOSE 7860

ENTRYPOINT ["/app/sd-server"]

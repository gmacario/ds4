# DwarfStar (ds4) CUDA container.
#
# Multi-stage build: a CUDA "devel" image compiles the binaries, then only the
# binaries and the CUDA runtime libraries are copied into a slim "runtime"
# image. Model GGUF files are NEVER baked into the image (they range from tens
# to hundreds of GB); mount them at runtime under /models instead.
#
# Build (choose CUDA_ARCH for your GPU, see the table below):
#   docker build -t ds4:cuda --build-arg CUDA_ARCH=sm_90 .
#
# Run (requires the NVIDIA Container Toolkit on the host):
#   docker run --rm --gpus all -p 8000:8000 \
#       -v /path/to/gguf:/models ds4:cuda \
#       ds4-server --host 0.0.0.0 --port 8000 -m /models/ds4flash.gguf --ctx 32768
#
# CUDA_ARCH picks the nvcc -arch value. Common choices:
#   sm_89   Ada Lovelace   (L40S, L4, RTX 4090)
#   sm_90   Hopper         (H100, H200)                    [default]
#   sm_100  Blackwell DC   (B100/B200)                     needs CUDA >= 12.8
#   sm_120  Blackwell RTX  (RTX 50xx)                      needs CUDA >= 12.8
#   (empty) let nvcc pick its default arch; this is the DGX Spark / GB10 path
#           mirrored from `make cuda-spark`. Build with:
#             docker build -t ds4:spark --build-arg CUDA_ARCH= .
#
# A build embeds SASS for the chosen arch plus forward-compatible PTX, so the
# image also runs on newer GPUs of the same family via JIT.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# nvcc -arch value; empty means "use nvcc default" (the DGX Spark path).
ARG CUDA_ARCH=sm_90
# CPU tuning for the C sources. -march=native ties the binary to the build
# machine's CPU; override to e.g. -march=x86-64-v3 for a portable image.
ARG CPU_FLAG=-march=native

COPY . .

# Build all five CUDA binaries. This mirrors what `make cuda-generic` /
# `make cuda-spark` do internally, but with an explicit, GPU-less arch so the
# image builds on hosts without a GPU attached.
RUN make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
        CUDA_ARCH="${CUDA_ARCH}" \
        NATIVE_CPU_FLAG="${CPU_FLAG}"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# curl + CA certs let the curl download path in download_model.sh fetch the
# smaller DeepSeek Flash GGUF files from inside the container. The larger PRO
# and GLM files still need the Hugging Face CLI, which is not installed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH=/app:${PATH}

COPY --from=builder /src/ds4 /src/ds4-server /src/ds4-bench /src/ds4-eval /src/ds4-agent /app/
COPY --from=builder /src/download_model.sh /app/

# Default model location for the CMD below; mount your GGUF here.
VOLUME ["/models"]
EXPOSE 8000

# ds4-server defaults to 127.0.0.1:8000, which is unreachable from outside the
# container, so bind 0.0.0.0 here. CUDA is already the default backend in a
# CUDA build. Override the whole command to run the CLI or a different model:
#   docker run --gpus all ds4:cuda ds4 -p "Hello" -m /models/ds4flash.gguf
CMD ["ds4-server", "--host", "0.0.0.0", "--port", "8000", \
     "-m", "/models/ds4flash.gguf", "--ctx", "32768"]

# Running DwarfStar (ds4) in a container

This describes how to build and run ds4 in a Docker container on
**CUDA-enabled hardware**. The container compiles the CUDA build of ds4 and
runs the OpenAI/Anthropic-compatible `ds4-server` against a model it downloads
into a mounted volume — model weights are **never** baked into the image, since
DeepSeek V4 Flash/PRO and GLM GGUF files range from tens to hundreds of GB.

## Prerequisites

On the host:

* An NVIDIA GPU with a recent driver.
* Docker with the **NVIDIA Container Toolkit** installed and configured, so
  `--gpus all` works. Verify with:

  ```sh
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
  ```

* Enough disk and RAM/VRAM for the model you intend to run (see the model sizes
  in `download_model.sh` and the README).

## Quick start

```sh
cp .env.example .env    # adjust DS4_MODEL, CUDA_ARCH, etc.
docker compose up --build
```

On first start the container downloads `DS4_MODEL` (default `q2-imatrix`, about
81 GB) into `./gguf` and starts `ds4-server` on `http://localhost:8000`.
Subsequent starts reuse the already-downloaded weights.

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds4","messages":[{"role":"user","content":"Hello"}]}'
```

## How the entrypoint works

The image's `ENTRYPOINT` is `entrypoint.sh`. Its behavior depends on the first
argument:

* **No arguments, or arguments starting with `-`** (i.e. `ds4-server` flags):
  the managed flow runs — download `DS4_MODEL` if needed, then start
  `ds4-server` configured from the environment variables below, with any flags
  you passed appended so they can override the generated ones.
* **An explicit command** (`ds4`, `ds4-bench`, `ds4-eval`, `ds4-agent`,
  `download_model.sh`, `sh`, ...): run directly, bypassing the managed flow
  entirely. Useful for the CLI, benchmarks, or manual model management:

  ```sh
  docker run --rm --gpus all -v "$PWD/gguf:/models" ds4:cuda \
      ds4 -p "Write a haiku about GPUs" -m /models/ds4flash.gguf

  docker run --rm --gpus all -v "$PWD/gguf:/models" ds4:cuda \
      ds4-bench -m /models/ds4flash.gguf --ctx 32768
  ```

  CUDA is the default backend in a CUDA build, so `--cuda` is not required.

## Environment variables

Set these with `docker run -e VAR=value`, in `.env` for Compose, or in
`docker-compose.yml`'s `environment:` block. See `.env.example` for the same
list with defaults inline.

| Variable | Default | Effect |
| --- | --- | --- |
| `DS4_MODEL` | `q2-imatrix` | Target passed to `download_model.sh` on startup. One of its supported names (`q2-imatrix`, `q2-q4-imatrix`, `q4-imatrix`, `pro-q2-imatrix`, `glm-unsloth-q4`, `glm-antirez-q2`, `glm-antirez-iq2xxs`, `glm-antirez-q4`, ...), or `none` to skip downloading and rely on `DS4_MODEL_PATH` or a pre-mounted `./gguf/ds4flash.gguf`. |
| `DS4_GGUF_DIR` | `/models` | Download directory inside the container; matches the `/models` volume. |
| `DS4_MODEL_PATH` | *(unset)* | Explicit `-m/--model` override, e.g. `/models/my-custom.gguf`. |
| `DS4_CTX` | `32768` | `--ctx`. |
| `DS4_HOST` | `0.0.0.0` | `--host`. `ds4-server` defaults to `127.0.0.1`, which is unreachable from outside the container; keep this at `0.0.0.0`. |
| `DS4_PORT` | `8000` | `--port`, and the port published by Compose. |
| `DS4_KV_DISK_DIR` | `/kv-cache` | `--kv-disk-dir`; matches the `/kv-cache` volume. Set to `none` to disable the on-disk KV cache. |
| `DS4_KV_DISK_SPACE_MB` | `8192` | `--kv-disk-space-mb`. |
| `DS4_ENABLE_MTP` | `0` | Set to `1`/`true`/`yes`/`on` to download the MTP GGUF and pass `--mtp`/`--mtp-draft`. Only compatible with the Flash q2/q4 imatrix quants. |
| `DS4_MTP_DRAFT` | `2` | `--mtp-draft`, used only when MTP is enabled. |
| `DS4_MTP_MARGIN` | *(unset)* | `--mtp-margin`, used only when MTP is enabled and set. |
| `DS4_THREADS` | *(unset)* | `--threads`, if set. |
| `DS4_EXTRA_ARGS` | *(unset)* | Extra flags appended last, e.g. `"--quality --cuda-tensor-parallel"`. Word-split, so quote multiple flags in one string. |
| `HF_TOKEN` | *(unset)* | Forwarded to `download_model.sh`, needed for the larger PRO/GLM downloads. |

Compose-only (host-side, not passed into the container):

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDA_ARCH` | `sm_90` | Build-time `nvcc -arch` value (see the table below). |
| `DS4_WEIGHTS_HOST_DIR` | `./gguf` | Host directory mounted at `/models`, writable. Matches `download_model.sh`'s own default so weights fetched on the host are reused. |
| `DS4_KV_HOST_DIR` | `./kv-cache` | Host directory mounted at `/kv-cache`. |

## Building for your GPU

Pick `CUDA_ARCH` for your GPU:

| GPU family            | Examples                | `CUDA_ARCH`         |
| --------------------- | ------------------------ | ------------------- |
| Ampere datacenter     | A100, A30               | `sm_80`             |
| Ampere                | A40, A10, RTX 30xx      | `sm_86`             |
| Ada Lovelace          | L40S, L4, RTX 4090      | `sm_89`             |
| Hopper                | H100, H200              | `sm_90` *(default)* |
| Blackwell datacenter  | B100, B200              | `sm_100`            |
| Blackwell RTX         | RTX 50xx                | `sm_120`            |
| DGX Spark / GB10      | GB10                    | *(empty)*           |

```sh
# Hopper (default)
docker build -t ds4:cuda .

# Ampere (A40)
docker build -t ds4:cuda --build-arg CUDA_ARCH=sm_86 .

# Ada Lovelace (L40S / RTX 4090)
docker build -t ds4:cuda --build-arg CUDA_ARCH=sm_89 .

# DGX Spark / GB10 — mirrors `make cuda-spark`, no explicit -arch
docker build -t ds4:spark --build-arg CUDA_ARCH= .

# With Compose:
CUDA_ARCH=sm_86 docker compose build
```

Notes:

* Blackwell (`sm_100`/`sm_120`) and GB10 need a recent CUDA. Override the base
  image if needed, e.g. `--build-arg CUDA_VERSION=12.8.1` (or newer).
* The image compiles with `-march=native` by default, which tunes the CPU code
  for the **build** machine. If you build and run on different CPUs, build a
  portable binary with `--build-arg CPU_FLAG=-march=x86-64-v3`.
* The `nvcc` compile of `ds4_cuda.cu` is large; expect a multi-minute build and
  a few GB of RAM during compilation.
* A build embeds SASS for the chosen arch plus forward-compatible PTX, so the
  image also runs on newer GPUs of the same family via JIT.
* **A40 requires `CUDA_ARCH=sm_86`.** The default (`sm_90`, Hopper) targets a
  different, newer GPU family; PTX forward compatibility does not cover running
  Hopper-targeted code on an older Ampere card. It may still build and even run
  without an explicit CUDA error, but is unsupported and not guaranteed correct
  or performant — always set `sm_86` explicitly for A40.

## Manual `docker run`

Without Compose, mount both volumes explicitly (both must be writable — the
container downloads weights into `/models` and writes checkpoints into
`/kv-cache`):

```sh
docker run --rm --gpus all -p 8000:8000 \
    -v "$PWD/gguf:/models" \
    -v "$PWD/kv-cache:/kv-cache" \
    -e DS4_MODEL=q2-imatrix \
    ds4:cuda
```

## Multi-GPU

On a single host with several CUDA GPUs, ds4 can split DeepSeek V4 Flash across
them with tensor parallelism:

```sh
DS4_EXTRA_ARGS="--cuda-tensor-parallel" docker compose up --build
```

Expose only some GPUs with `--gpus '"device=0,1"'` or the standard
`NVIDIA_VISIBLE_DEVICES` environment variable. **DGX Spark / GB10 is a
single-GPU target and must not be started with `--cuda-tensor-parallel`.**

See the "Tensor Parallelism across CUDA GPUs" section of the README for tuning
details.

## Troubleshooting

* **`nvidia-smi` fails inside the container / "could not select device driver":**
  the NVIDIA Container Toolkit is not installed or configured on the host.
* **Illegal instruction / `SIGILL` at startup:** the image was built with
  `-march=native` on a different CPU than the one running it. Rebuild with
  `--build-arg CPU_FLAG=-march=x86-64-v3`.
* **`no kernel image is available for execution`:** the `CUDA_ARCH` used at
  build time does not match (and cannot JIT to) the runtime GPU. Rebuild with
  the correct arch from the table above.
* **Server reachable only from localhost:** ensure `DS4_HOST=0.0.0.0` (the
  default) and that you published the port.
* **"Unknown model" from `download_model.sh`:** `DS4_MODEL` must be one of its
  exact target names (`q2-imatrix`, not `q2`); see the table above or run
  `download_model.sh --help`.
* **Model re-downloads every start:** confirm the weights volume is actually
  persisted (bind mount or named volume), not the container's writable layer.

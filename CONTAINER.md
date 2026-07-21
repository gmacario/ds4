# Running DwarfStar (ds4) in a container

This describes how to build and run ds4 in a Docker container on
**CUDA-enabled hardware**. The container compiles the CUDA build of ds4 and
runs the OpenAI/Anthropic-compatible `ds4-server` (or any other ds4 binary)
against a model you mount from the host.

Model weights are **never** baked into the image. DeepSeek V4 Flash/PRO and GLM
GGUF files range from tens to hundreds of GB, so you mount them at runtime.

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

## 1. Get a model

Download a GGUF onto the host into a directory you will mount, for example
`./gguf`. `download_model.sh` creates the `ds4flash.gguf` symlink the defaults
expect:

```sh
./download_model.sh q2-imatrix        # ~81 GB, good for a single 96/128 GB GPU class
# or a larger quant on bigger machines:
./download_model.sh q4-imatrix        # ~153 GB
```

You can also run the downloader inside the container (curl path only, used by
the smaller Flash files):

```sh
docker run --rm -v "$PWD/gguf:/models" -e DS4_GGUF_DIR=/models \
    ds4:cuda download_model.sh q2-imatrix
```

The larger PRO and GLM files need the Hugging Face CLI, which is not installed
in the image; download those on the host instead.

## 2. Build the image

Pick `CUDA_ARCH` for your GPU:

| GPU family            | Examples                | `CUDA_ARCH`         |
| --------------------- | ----------------------- | ------------------- |
| Ada Lovelace          | L40S, L4, RTX 4090      | `sm_89`             |
| Hopper                | H100, H200              | `sm_90` *(default)* |
| Blackwell datacenter  | B100, B200              | `sm_100`            |
| Blackwell RTX         | RTX 50xx                | `sm_120`            |
| DGX Spark / GB10      | GB10                    | *(empty)*           |

```sh
# Hopper (default)
docker build -t ds4:cuda .

# Ada Lovelace (L40S / RTX 4090)
docker build -t ds4:cuda --build-arg CUDA_ARCH=sm_89 .

# DGX Spark / GB10 — mirrors `make cuda-spark`, no explicit -arch
docker build -t ds4:spark --build-arg CUDA_ARCH= .
```

Notes:

* Blackwell (`sm_100`/`sm_120`) and GB10 need a recent CUDA. Override the base
  image if needed, e.g. `--build-arg CUDA_VERSION=12.8.1` (or newer).
* The image compiles with `-march=native` by default, which tunes the CPU code
  for the **build** machine. If you build and run on different CPUs, build a
  portable binary with `--build-arg CPU_FLAG=-march=x86-64-v3`.
* The `nvcc` compile of `ds4_cuda.cu` is large; expect a multi-minute build and
  a few GB of RAM during compilation.

## 3. Run the server

```sh
docker run --rm --gpus all -p 8000:8000 \
    -v "$PWD/gguf:/models:ro" \
    ds4:cuda \
    ds4-server --host 0.0.0.0 --port 8000 -m /models/ds4flash.gguf --ctx 32768
```

The default `CMD` already runs exactly this, so if your model is at
`/models/ds4flash.gguf` you can shorten it to:

```sh
docker run --rm --gpus all -p 8000:8000 -v "$PWD/gguf:/models:ro" ds4:cuda
```

Then call the API (OpenAI-compatible):

```sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds4","messages":[{"role":"user","content":"Hello"}]}'
```

> The server binds `127.0.0.1:8000` by default, which is unreachable from
> outside the container. The image overrides this to `--host 0.0.0.0`; keep
> that flag if you write your own command.

## 4. Run the CLI or other binaries

The image ships all five binaries (`ds4`, `ds4-server`, `ds4-bench`,
`ds4-eval`, `ds4-agent`) on the `PATH`. Override the command to use them:

```sh
# One-shot CLI prompt
docker run --rm --gpus all -v "$PWD/gguf:/models:ro" ds4:cuda \
    ds4 -p "Write a haiku about GPUs" -m /models/ds4flash.gguf

# Benchmark
docker run --rm --gpus all -v "$PWD/gguf:/models:ro" ds4:cuda \
    ds4-bench -m /models/ds4flash.gguf --ctx 32768
```

CUDA is the default backend in a CUDA build, so `--cuda` is not required.

## 5. Docker Compose

`docker-compose.yml` wires up the build, the GPU reservation, the model volume,
and the server command:

```sh
# Build for your GPU and start the server
CUDA_ARCH=sm_90 DS4_GGUF_DIR=./gguf docker compose up --build
```

Environment variables it honors:

* `CUDA_ARCH` — GPU arch passed to the build (default `sm_90`).
* `DS4_GGUF_DIR` — host directory mounted at `/models` (default `./gguf`).
* `DS4_MODEL` — model path inside the container (default
  `/models/ds4flash.gguf`).

## Multi-GPU

On a single host with several CUDA GPUs, ds4 can split DeepSeek V4 Flash across
them with tensor parallelism. Add the flag to the command:

```sh
docker run --rm --gpus all -p 8000:8000 -v "$PWD/gguf:/models:ro" ds4:cuda \
    ds4-server --host 0.0.0.0 --port 8000 -m /models/ds4flash.gguf \
    --ctx 32768 --cuda-tensor-parallel
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
* **Server reachable only from localhost:** ensure the command includes
  `--host 0.0.0.0` and you published the port with `-p 8000:8000`.
* **Model not found:** confirm the GGUF is under the mounted directory and the
  `-m` path matches, e.g. `/models/ds4flash.gguf`.

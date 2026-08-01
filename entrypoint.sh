#!/bin/sh
# Container entrypoint for ds4-server.
#
# If the first argument is an explicit command (a binary name, a script, a
# shell), it is executed directly so `docker run ds4:cuda ds4 -p "hi" -m ...`
# and similar keep working exactly like running the binaries outside a
# container. Otherwise (no arguments, or arguments that are ds4-server flags)
# this script downloads the configured model if needed and starts ds4-server,
# configured from environment variables so the common case needs no arguments
# at all.
set -eu

case "${1:-}" in
    ""|-*) ;;
    *) exec "$@" ;;
esac

DS4_HOST=${DS4_HOST:-0.0.0.0}
DS4_PORT=${DS4_PORT:-8000}
DS4_CTX=${DS4_CTX:-32768}
DS4_MODEL=${DS4_MODEL:-q2-imatrix}
DS4_GGUF_DIR=${DS4_GGUF_DIR:-/models}
DS4_MODEL_PATH=${DS4_MODEL_PATH:-}
DS4_KV_DISK_DIR=${DS4_KV_DISK_DIR:-/kv-cache}
DS4_KV_DISK_SPACE_MB=${DS4_KV_DISK_SPACE_MB:-8192}
DS4_ENABLE_MTP=${DS4_ENABLE_MTP:-0}
DS4_MTP_DRAFT=${DS4_MTP_DRAFT:-2}
DS4_MTP_MARGIN=${DS4_MTP_MARGIN:-}
DS4_THREADS=${DS4_THREADS:-}
DS4_EXTRA_ARGS=${DS4_EXTRA_ARGS:-}

export DS4_GGUF_DIR

mkdir -p "$DS4_GGUF_DIR"
if [ -n "$DS4_KV_DISK_DIR" ] && [ "$DS4_KV_DISK_DIR" != "none" ]; then
    mkdir -p "$DS4_KV_DISK_DIR"
fi

if [ -n "$DS4_MODEL" ] && [ "$DS4_MODEL" != "none" ]; then
    /app/download_model.sh "$DS4_MODEL"
fi

# Base flags first, then any user-supplied flags from "$@": ds4-server's
# argument parser applies flags left to right, so later duplicates win and a
# manually passed override always beats the env-var-derived default.
set -- \
    --host "$DS4_HOST" \
    --port "$DS4_PORT" \
    --ctx "$DS4_CTX" \
    "$@"

if [ -n "$DS4_KV_DISK_DIR" ] && [ "$DS4_KV_DISK_DIR" != "none" ]; then
    set -- "$@" --kv-disk-dir "$DS4_KV_DISK_DIR" --kv-disk-space-mb "$DS4_KV_DISK_SPACE_MB"
fi

case "$DS4_ENABLE_MTP" in
    1|true|TRUE|yes|YES|on|ON)
        /app/download_model.sh mtp
        # download_model.sh names the MTP file after its quant; glob for it
        # instead of hardcoding the name so a renamed release doesn't break this.
        mtp_file=$(ls -t "$DS4_GGUF_DIR"/*MTP*.gguf 2>/dev/null | head -n1)
        if [ -z "$mtp_file" ]; then
            echo "ds4-entrypoint: DS4_ENABLE_MTP is set but no MTP GGUF was found in $DS4_GGUF_DIR" >&2
            exit 1
        fi
        set -- "$@" --mtp "$mtp_file" --mtp-draft "$DS4_MTP_DRAFT"
        if [ -n "$DS4_MTP_MARGIN" ]; then
            set -- "$@" --mtp-margin "$DS4_MTP_MARGIN"
        fi
        ;;
esac

if [ -n "$DS4_MODEL_PATH" ]; then
    set -- "$@" --model "$DS4_MODEL_PATH"
fi

if [ -n "$DS4_THREADS" ]; then
    set -- "$@" --threads "$DS4_THREADS"
fi

if [ -n "$DS4_EXTRA_ARGS" ]; then
    # Intentionally word-split so DS4_EXTRA_ARGS can carry multiple flags.
    # shellcheck disable=SC2086
    set -- "$@" $DS4_EXTRA_ARGS
fi

exec /app/ds4-server "$@"

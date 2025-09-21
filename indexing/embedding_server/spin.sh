#!/usr/bin/env bash
set -euo pipefail
set -a
. ".env" 
set +a   

model=Qwen/Qwen3-Embedding-0.6B
volume=tei-cache
DTYPE='float16'

docker run --gpus all -p 8080:80 -v $volume:/data \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-embeddings-inference:86-1.8 \
  --model-id $model \
  --dtype $DTYPE \
  --max-batch-tokens 16384 \
  --max-batch-requests 32 \
  --default-prompt-name query \
  --prometheus-port 9000 \
  --auto-truncate \
#   --json-output



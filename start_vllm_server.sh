#!/bin/bash

# Configuration
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct" # Specify your desired model here
UDS_PATH="/tmp/vllm-server.sock"

echo "Starting vLLM OpenAI-compatible server..."
echo "Model: $MODEL_ID"
echo "UDS Socket: $UDS_PATH"
echo "Note: This server will run in the foreground. Press Ctrl+C to stop it."

# Clean up old socket if it exists
rm -f $UDS_PATH

# Start the server
# --uds: Use Unix Domain Socket for IPC (faster than TCP loopback)
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --uds $UDS_PATH

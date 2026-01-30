#!/bin/bash
# M4: Bedrock Integration Test
# Run on PARENT EC2 instance

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "M4: BEDROCK VIA ENCLAVE TEST"
echo "============================"

# Cleanup
nitro-cli terminate-enclave --all 2>/dev/null || true

# Build
cd "$SCRIPT_DIR"
docker build -t isol8-enclave:latest -f Dockerfile.enclave .
nitro-cli build-enclave --docker-uri isol8-enclave:latest --output-file enclave.eif

# Start vsock-proxy
python3 "$SCRIPT_DIR/vsock_proxy.py" &
PROXY_PID=$!
trap "kill $PROXY_PID 2>/dev/null; nitro-cli terminate-enclave --all 2>/dev/null" EXIT
sleep 2

# Run enclave
ENCLAVE_OUTPUT=$(nitro-cli run-enclave --cpu-count 2 --memory 512 --eif-path enclave.eif --debug-mode)
echo "$ENCLAVE_OUTPUT"
ENCLAVE_CID=$(echo "$ENCLAVE_OUTPUT" | grep -o '"EnclaveCID": [0-9]*' | grep -o '[0-9]*')
echo "Enclave CID: $ENCLAVE_CID"
sleep 5

# Test
python3 "$SCRIPT_DIR/test_bedrock_client.py" "$ENCLAVE_CID"

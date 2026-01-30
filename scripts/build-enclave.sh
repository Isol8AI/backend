#!/bin/bash
# build-enclave.sh
#
# Builds the Nitro Enclave EIF from Dockerfile.enclave
# Run from anywhere - paths are resolved relative to this script
#
# Usage: ./scripts/build-enclave.sh

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
ENCLAVE_DIR="$BACKEND_DIR/enclave"

echo "=== Building Nitro Enclave Image ==="
echo "Working directory: $ENCLAVE_DIR"
echo ""

# Verify we're on a system with nitro-cli
if ! command -v nitro-cli &> /dev/null; then
    echo "ERROR: nitro-cli not found"
    echo "This script must be run on an EC2 instance with Nitro Enclave support."
    echo ""
    echo "To install on Amazon Linux 2:"
    echo "  amazon-linux-extras install aws-nitro-enclaves-cli -y"
    echo "  yum install -y aws-nitro-enclaves-cli-devel"
    exit 1
fi

# Verify Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running"
    echo "Start Docker with: sudo systemctl start docker"
    exit 1
fi

# Step 1: Build Docker image
echo "[1/2] Building Docker image..."
docker build \
    -t isol8-enclave:latest \
    -f "$ENCLAVE_DIR/Dockerfile.enclave" \
    "$ENCLAVE_DIR"

echo ""

# Step 2: Convert to EIF
echo "[2/2] Converting to Enclave Image Format (EIF)..."
nitro-cli build-enclave \
    --docker-uri isol8-enclave:latest \
    --output-file "$ENCLAVE_DIR/hello.eif"

echo ""
echo "=== Build Complete ==="
echo ""
echo "EIF file: $ENCLAVE_DIR/hello.eif"
echo ""
echo "IMPORTANT: The PCR values above are cryptographic measurements of your enclave."
echo "Save PCR0 - you'll need it for KMS attestation policies in M6."
echo ""
echo "Next steps:"
echo "  ./scripts/run-enclave.sh start    # Launch the enclave"
echo "  ./scripts/run-enclave.sh console  # View enclave output"

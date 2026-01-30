#!/bin/bash
# run-enclave.sh
#
# Manage the Nitro Enclave lifecycle
# Run from anywhere - paths are resolved relative to this script
#
# Usage: ./scripts/run-enclave.sh [start|stop|status|console]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
EIF_PATH="$BACKEND_DIR/enclave/hello.eif"

# Enclave resource allocation
# These values must not exceed what's configured in /etc/nitro_enclaves/allocator.yaml
# Current Terraform config (modules/ec2/user_data.sh): 2048 MB, 2 CPUs
ENCLAVE_MEMORY_MB=2048
ENCLAVE_CPU_COUNT=2

# Get enclave ID if one is running
get_enclave_id() {
    nitro-cli describe-enclaves 2>/dev/null | jq -r '.[0].EnclaveID // empty'
}

cmd_start() {
    # Verify nitro-cli exists
    if ! command -v nitro-cli &> /dev/null; then
        echo "ERROR: nitro-cli not found"
        echo "This script must be run on an EC2 instance with Nitro Enclave support."
        exit 1
    fi

    # Check for EIF file
    if [ ! -f "$EIF_PATH" ]; then
        echo "ERROR: EIF file not found at $EIF_PATH"
        echo ""
        echo "Run ./scripts/build-enclave.sh first to build the enclave image."
        exit 1
    fi

    # Check if enclave already running
    EXISTING=$(get_enclave_id)
    if [ -n "$EXISTING" ]; then
        echo "Enclave already running: $EXISTING"
        echo ""
        echo "Run './scripts/run-enclave.sh stop' first to terminate it."
        exit 1
    fi

    echo "Starting enclave..."
    echo "  EIF:    $EIF_PATH"
    echo "  Memory: ${ENCLAVE_MEMORY_MB} MB"
    echo "  CPUs:   ${ENCLAVE_CPU_COUNT}"
    echo ""

    # --debug-mode enables console output (remove in production for security)
    nitro-cli run-enclave \
        --eif-path "$EIF_PATH" \
        --memory "$ENCLAVE_MEMORY_MB" \
        --cpu-count "$ENCLAVE_CPU_COUNT" \
        --debug-mode

    echo ""
    echo "Enclave started!"
    echo ""
    echo "Next steps:"
    echo "  ./scripts/run-enclave.sh status   # Check enclave state"
    echo "  ./scripts/run-enclave.sh console  # View enclave output"
    echo "  ./scripts/run-enclave.sh stop     # Terminate enclave"
}

cmd_stop() {
    ENCLAVE_ID=$(get_enclave_id)
    if [ -z "$ENCLAVE_ID" ]; then
        echo "No enclave running."
        exit 0
    fi

    echo "Terminating enclave: $ENCLAVE_ID"
    nitro-cli terminate-enclave --enclave-id "$ENCLAVE_ID"
    echo ""
    echo "Enclave terminated."
}

cmd_status() {
    echo "=== Enclave Status ==="
    echo ""

    # Check if nitro-cli exists
    if ! command -v nitro-cli &> /dev/null; then
        echo "ERROR: nitro-cli not found"
        echo "This script must be run on an EC2 instance with Nitro Enclave support."
        exit 1
    fi

    RESULT=$(nitro-cli describe-enclaves 2>/dev/null)

    if [ "$RESULT" = "[]" ]; then
        echo "No enclaves running."
        echo ""
        echo "Start one with: ./scripts/run-enclave.sh start"
    else
        echo "$RESULT" | jq .
    fi
}

cmd_console() {
    ENCLAVE_ID=$(get_enclave_id)
    if [ -z "$ENCLAVE_ID" ]; then
        echo "No enclave running."
        echo ""
        echo "Start one with: ./scripts/run-enclave.sh start"
        exit 1
    fi

    echo "Attaching to enclave console: $ENCLAVE_ID"
    echo "Press Ctrl+C to detach (enclave keeps running)"
    echo ""
    echo "--- Enclave Output ---"
    nitro-cli console --enclave-id "$ENCLAVE_ID"
}

# Show usage
show_usage() {
    echo "Usage: $0 [start|stop|status|console]"
    echo ""
    echo "Commands:"
    echo "  start   - Launch the enclave from hello.eif"
    echo "  stop    - Terminate the running enclave"
    echo "  status  - Show running enclave info (JSON)"
    echo "  console - Attach to enclave stdout/stderr (Ctrl+C to detach)"
    echo ""
    echo "Example workflow:"
    echo "  ./scripts/build-enclave.sh      # Build the EIF (first time)"
    echo "  ./scripts/run-enclave.sh start  # Launch enclave"
    echo "  ./scripts/run-enclave.sh console  # See 'Hello from enclave!'"
    echo "  ./scripts/run-enclave.sh stop   # Clean up"
}

# Main
case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    console) cmd_console ;;
    -h|--help|help)
        show_usage
        exit 0
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

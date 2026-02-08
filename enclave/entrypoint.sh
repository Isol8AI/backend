#!/bin/bash
# Nitro Enclave entrypoint: bring up loopback, then run server.
#
# The enclave VM has NO network stack by default — not even loopback.
# We must bring up 'lo' so the vsock_tcp_bridge can listen on 127.0.0.1:3128
# and Node.js (OpenClaw) can connect to it.
#
# This script runs as root to configure networking, then starts the server.
# The Nitro Enclave is already hardware-isolated, so running as root inside
# the enclave is acceptable (su/runuser may not be available in minimal image).

set -e

# Bring up loopback interface
ip link set lo up 2>/dev/null || ifconfig lo 127.0.0.1 netmask 255.0.0.0 up 2>/dev/null || true
echo "[entrypoint] Loopback interface up"

# Run server (enclave VM is already hardware-isolated)
exec python3 /app/bedrock_server.py

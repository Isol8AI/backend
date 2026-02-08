#!/bin/bash
# Nitro Enclave entrypoint: bring up loopback, then run server as enclave user.
#
# The enclave VM has NO network stack by default — not even loopback.
# We must bring up 'lo' so the vsock_tcp_bridge can listen on 127.0.0.1:3128
# and Node.js (OpenClaw) can connect to it.
#
# This script runs as root (to configure networking), then drops to the
# 'enclave' user for the actual server process.

set -e

# Bring up loopback interface
ip link set lo up 2>/dev/null || ifconfig lo 127.0.0.1 netmask 255.0.0.0 up 2>/dev/null || true
echo "[entrypoint] Loopback interface up"

# Drop privileges and run server
exec su -s /bin/bash enclave -c "python3 /app/bedrock_server.py"

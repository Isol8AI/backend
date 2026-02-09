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

# Verify ip command exists (iproute package must be in Dockerfile)
if ! command -v ip &>/dev/null; then
    echo "[entrypoint] FATAL: 'ip' command not found. Install iproute in Dockerfile." >&2
    exit 1
fi

# Bring up loopback interface with full configuration.
# On the Nitro Enclave kernel (4.14), `ip link set lo up` assigns the address
# but does NOT add the local route. bind() works without a route, but connect()
# returns ENETUNREACH. We must explicitly add the address and route.
ip link set lo up
ip addr add 127.0.0.1/8 dev lo 2>/dev/null || true
ip route add local 127.0.0.0/8 dev lo table local 2>/dev/null || true

# Verify loopback is fully configured (address + route)
echo "[entrypoint] Loopback config:"
ip addr show lo
ip route show table local | grep 127 || echo "[entrypoint] WARNING: no local route for 127"

# Run server (enclave VM is already hardware-isolated)
exec python3 /app/bedrock_server.py

#!/usr/bin/env python3
"""
Minimal enclave application for M1: Hello Enclave.

This validates that we can build and run a Nitro Enclave.
It prints a startup message and periodic heartbeats to the debug console.

Run with: ./scripts/run-enclave.sh start
View output: ./scripts/run-enclave.sh console
"""

import sys
import time


def main():
    # Flush stdout immediately - enclave console buffers aggressively
    print("=" * 50, flush=True)
    print("HELLO FROM INSIDE THE NITRO ENCLAVE!", flush=True)
    print("=" * 50, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print("Enclave started successfully.", flush=True)
    print("", flush=True)

    # Keep running with heartbeats
    counter = 0
    while True:
        counter += 1
        print(f"[Heartbeat #{counter}] Enclave is alive...", flush=True)
        time.sleep(5)  # Print every 5 seconds


if __name__ == "__main__":
    main()

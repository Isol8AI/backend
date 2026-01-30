#!/usr/bin/env python3
"""
M4: HTTP-over-vsock Client for Nitro Enclave
=============================================

This module provides HTTP/HTTPS client capabilities for code running inside
a Nitro Enclave. Since enclaves have no direct network access, all HTTP
requests are tunneled through vsock-proxy on the parent instance.

Key security property: TLS termination happens HERE, inside the enclave.
The parent proxy only sees encrypted TLS bytes.

Usage:
    from vsock_http_client import VsockHttpClient

    client = VsockHttpClient()
    response = client.request(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/...",
        headers={"Content-Type": "application/json"},
        body=b'{"prompt": "Hello"}'
    )
    print(response.status, response.body)
"""

import socket
import ssl
import json
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

# vsock constants
AF_VSOCK = 40
VMADDR_CID_HOST = 3  # Parent's CID from enclave perspective
VSOCK_PROXY_PORT = 8443


@dataclass
class HttpResponse:
    """HTTP response container."""

    status: int
    status_text: str
    headers: Dict[str, str]
    body: bytes

    def json(self) -> dict:
        """Parse body as JSON."""
        return json.loads(self.body.decode("utf-8"))


class VsockHttpClient:
    """
    HTTP client that tunnels through vsock-proxy.

    The flow is:
    1. Connect to parent via vsock (CID 3, port 8443)
    2. Send HTTP CONNECT request for the target host
    3. Receive 200 OK from proxy
    4. Wrap socket with TLS (termination happens here in enclave)
    5. Send actual HTTP request over TLS
    6. Receive response
    """

    def __init__(self, proxy_cid: int = VMADDR_CID_HOST, proxy_port: int = VSOCK_PROXY_PORT):
        self.proxy_cid = proxy_cid
        self.proxy_port = proxy_port

    def _create_vsock(self) -> socket.socket:
        """Create a vsock connection to the proxy."""
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect((self.proxy_cid, self.proxy_port))
        return sock

    def _send_connect(self, sock: socket.socket, host: str, port: int) -> bool:
        """
        Send HTTP CONNECT request to proxy.
        Returns True if proxy accepts the connection.
        """
        connect_request = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}\r\n\r\n"
        sock.sendall(connect_request.encode("utf-8"))

        # Read response (should be "HTTP/1.1 200 Connection Established")
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("Proxy closed connection")
            response += chunk

        # Check for 200 status
        status_line = response.split(b"\r\n")[0].decode("utf-8")
        return "200" in status_line

    def _wrap_tls(self, sock: socket.socket, host: str) -> ssl.SSLSocket:
        """
        Wrap socket with TLS. This is where TLS termination happens.
        The plaintext exists only inside this enclave.
        """
        context = ssl.create_default_context()
        # Verify certificates for security
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context.wrap_socket(sock, server_hostname=host)

    def _build_request(
        self,
        method: str,
        path: str,
        host: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None,
    ) -> bytes:
        """Build HTTP/1.1 request."""
        lines = [f"{method} {path} HTTP/1.1"]
        lines.append(f"Host: {host}")

        for key, value in headers.items():
            lines.append(f"{key}: {value}")

        if body:
            lines.append(f"Content-Length: {len(body)}")

        lines.append("")  # Empty line before body
        lines.append("")

        request = "\r\n".join(lines).encode("utf-8")
        if body:
            request = request + body

        return request

    def _parse_response(self, sock: ssl.SSLSocket) -> HttpResponse:
        """Parse HTTP response from socket."""
        # Read headers
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed while reading headers")
            header_data += chunk

        # Split headers and any body data received
        header_part, body_start = header_data.split(b"\r\n\r\n", 1)

        # Parse status line and headers
        lines = header_part.decode("utf-8").split("\r\n")
        status_line = lines[0]
        # Parse: "HTTP/1.1 200 OK"
        parts = status_line.split(" ", 2)
        status = int(parts[1])
        status_text = parts[2] if len(parts) > 2 else ""

        headers = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value

        # Read body based on Content-Length or chunked encoding
        content_length = int(headers.get("content-length", 0))

        body = body_start
        while len(body) < content_length:
            remaining = content_length - len(body)
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                break
            body += chunk

        return HttpResponse(
            status=status,
            status_text=status_text,
            headers=headers,
            body=body,
        )

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> HttpResponse:
        """
        Make an HTTP/HTTPS request through the vsock proxy.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL (e.g., https://api.example.com/path)
            headers: Request headers
            body: Request body (bytes)

        Returns:
            HttpResponse object
        """
        headers = headers or {}

        # Parse URL
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        # Determine port
        port = 443 if parsed.scheme == "https" else 80
        if ":" in host:
            host, port = host.rsplit(":", 1)
            port = int(port)

        use_tls = parsed.scheme == "https"

        # Connect through proxy
        sock = self._create_vsock()

        try:
            # Send CONNECT to proxy
            if not self._send_connect(sock, host, port):
                raise ConnectionError("Proxy refused connection")

            # Wrap with TLS if needed (this is the key security property!)
            if use_tls:
                sock = self._wrap_tls(sock, host)

            # Build and send request
            request = self._build_request(method, path, host, headers, body)
            sock.sendall(request)

            # Read response
            response = self._parse_response(sock)

            return response

        finally:
            sock.close()


# Convenience function for simple requests
def vsock_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> HttpResponse:
    """Make a single HTTP request through vsock proxy."""
    client = VsockHttpClient()
    return client.request(method, url, headers, body)


if __name__ == "__main__":
    # Quick test (run inside enclave with proxy running)
    print("Testing vsock HTTP client...")

    client = VsockHttpClient()

    # This would only work if vsock-proxy is running on parent
    # and the enclave has network connectivity through it
    print("Client initialized. Use client.request() to make requests.")

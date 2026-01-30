#!/usr/bin/env python3
"""
M4: IMDS Credential Helper for Parent Instance
================================================

This module runs on the PARENT EC2 instance (not in the enclave) to retrieve
temporary AWS credentials from the Instance Metadata Service (IMDS).

The EC2 instance has an IAM role attached. IMDS provides temporary credentials
that the parent can pass to the enclave via vsock.

Usage (on parent instance):
    from imds_credentials import get_iam_credentials

    creds = get_iam_credentials()
    # creds = {
    #     "access_key_id": "ASIA...",
    #     "secret_access_key": "...",
    #     "session_token": "...",
    #     "expiration": "2024-01-01T12:00:00Z"
    # }
"""

import json
import urllib.request
from typing import Dict, Optional


# IMDSv2 endpoints
IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_ROLE_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


def _get_imds_token(ttl_seconds: int = 21600) -> str:
    """
    Get IMDSv2 token.

    IMDSv2 requires a session token for security.
    Default TTL is 6 hours (21600 seconds).
    """
    request = urllib.request.Request(
        IMDS_TOKEN_URL,
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": str(ttl_seconds)},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode("utf-8")


def _imds_get(url: str, token: str) -> str:
    """Make an IMDSv2 GET request."""
    request = urllib.request.Request(
        url,
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode("utf-8")


def get_iam_role_name(token: Optional[str] = None) -> str:
    """
    Get the IAM role name attached to this instance.
    """
    if not token:
        token = _get_imds_token()

    role_name = _imds_get(IMDS_ROLE_URL, token)
    return role_name.strip()


def get_iam_credentials(role_name: Optional[str] = None) -> Dict[str, str]:
    """
    Get temporary IAM credentials from IMDS.

    Args:
        role_name: IAM role name. If not provided, will be auto-detected.

    Returns:
        Dict with:
            - access_key_id: AWS access key ID
            - secret_access_key: AWS secret access key
            - session_token: Session token (required for SigV4)
            - expiration: Credential expiration timestamp

    Raises:
        Exception: If credentials cannot be retrieved
    """
    token = _get_imds_token()

    if not role_name:
        role_name = get_iam_role_name(token)

    # Get credentials for the role
    creds_url = f"{IMDS_ROLE_URL}{role_name}"
    creds_json = _imds_get(creds_url, token)
    creds_data = json.loads(creds_json)

    return {
        "access_key_id": creds_data["AccessKeyId"],
        "secret_access_key": creds_data["SecretAccessKey"],
        "session_token": creds_data["Token"],
        "expiration": creds_data["Expiration"],
    }


def get_instance_region() -> str:
    """
    Get the region this instance is running in.
    """
    token = _get_imds_token()
    az = _imds_get("http://169.254.169.254/latest/meta-data/placement/availability-zone", token)
    # Region is AZ without the last character (e.g., us-east-1a -> us-east-1)
    return az.strip()[:-1]


if __name__ == "__main__":
    print("=" * 60)
    print("IMDS CREDENTIAL HELPER (M4)")
    print("=" * 60)

    try:
        region = get_instance_region()
        print(f"Instance region: {region}")

        role = get_iam_role_name()
        print(f"IAM role: {role}")

        creds = get_iam_credentials(role)
        print(f"Access Key ID: {creds['access_key_id'][:10]}...")
        print(f"Expiration: {creds['expiration']}")
        print("\n[SUCCESS] Credentials retrieved from IMDS")

    except urllib.error.URLError as e:
        print(f"\n[ERROR] Cannot reach IMDS: {e}")
        print("This script must run on an EC2 instance with an IAM role attached.")
    except Exception as e:
        print(f"\n[ERROR] {e}")

"""Tests for Pydantic response models."""
import pytest

from schemas.user_schemas import SyncUserResponse, UserPublicKeyResponse, CreateKeysResponse
from schemas.chat import EnclaveHealthResponse, EncryptionCheckResponse
from schemas.organization_encryption import (
    CreateOrgKeysResponse,
    DistributeOrgKeyResponse,
    AdminRecoveryKeyResponse,
)
from schemas.agent import AgentStateResponse


class TestSyncUserResponse:
    def test_created(self):
        resp = SyncUserResponse(status="created", user_id="user_123")
        assert resp.status == "created"
        assert resp.user_id == "user_123"

    def test_exists(self):
        resp = SyncUserResponse(status="exists", user_id="user_456")
        assert resp.status == "exists"
        assert resp.user_id == "user_456"

    def test_serialization(self):
        resp = SyncUserResponse(status="created", user_id="user_123")
        data = resp.model_dump()
        assert data == {"status": "created", "user_id": "user_123"}


class TestUserPublicKeyResponse:
    def test_valid(self):
        pub_key = "aa" * 32
        resp = UserPublicKeyResponse(user_id="user_123", public_key=pub_key)
        assert resp.user_id == "user_123"
        assert resp.public_key == pub_key

    def test_serialization(self):
        pub_key = "bb" * 32
        resp = UserPublicKeyResponse(user_id="user_789", public_key=pub_key)
        data = resp.model_dump()
        assert data == {"user_id": "user_789", "public_key": pub_key}


class TestCreateKeysResponse:
    def test_valid(self):
        pub_key = "aa" * 32
        resp = CreateKeysResponse(status="created", public_key=pub_key)
        assert resp.status == "created"
        assert resp.public_key == pub_key

    def test_serialization(self):
        pub_key = "cc" * 32
        resp = CreateKeysResponse(status="created", public_key=pub_key)
        data = resp.model_dump()
        assert data == {"status": "created", "public_key": pub_key}


class TestEnclaveHealthResponse:
    def test_healthy(self):
        resp = EnclaveHealthResponse(status="healthy", enclave_type="nitro")
        assert resp.status == "healthy"
        assert resp.enclave_type == "nitro"
        assert resp.enclave_public_key is None

    def test_with_public_key(self):
        pub_key = "dd" * 32
        resp = EnclaveHealthResponse(status="healthy", enclave_type="mock", enclave_public_key=pub_key)
        assert resp.enclave_public_key == pub_key

    def test_unhealthy(self):
        resp = EnclaveHealthResponse(status="unhealthy", enclave_type="nitro")
        assert resp.status == "unhealthy"

    def test_serialization(self):
        resp = EnclaveHealthResponse(status="healthy", enclave_type="nitro")
        data = resp.model_dump()
        assert data == {"status": "healthy", "enclave_type": "nitro", "enclave_public_key": None}


class TestEncryptionCheckResponse:
    def test_can_send(self):
        resp = EncryptionCheckResponse(can_send_encrypted=True, context="personal")
        assert resp.can_send_encrypted is True
        assert resp.error is None
        assert resp.org_id is None

    def test_cannot_send(self):
        resp = EncryptionCheckResponse(
            can_send_encrypted=False,
            error="No encryption keys set up",
            context="personal",
        )
        assert resp.can_send_encrypted is False
        assert resp.error == "No encryption keys set up"

    def test_with_org(self):
        resp = EncryptionCheckResponse(can_send_encrypted=True, context="organization", org_id="org_456")
        assert resp.org_id == "org_456"

    def test_serialization(self):
        resp = EncryptionCheckResponse(can_send_encrypted=True, context="personal")
        data = resp.model_dump()
        assert data == {"can_send_encrypted": True, "error": None, "context": "personal", "org_id": None}


class TestCreateOrgKeysResponse:
    def test_valid(self):
        resp = CreateOrgKeysResponse(status="created", org_id="org_123")
        assert resp.status == "created"
        assert resp.org_id == "org_123"

    def test_serialization(self):
        resp = CreateOrgKeysResponse(status="created", org_id="org_123")
        data = resp.model_dump()
        assert data == {"status": "created", "org_id": "org_123"}


class TestDistributeOrgKeyResponse:
    def test_valid(self):
        resp = DistributeOrgKeyResponse(status="distributed", membership_id="mem_123")
        assert resp.status == "distributed"
        assert resp.membership_id == "mem_123"

    def test_serialization(self):
        resp = DistributeOrgKeyResponse(status="distributed", membership_id="mem_456")
        data = resp.model_dump()
        assert data == {"status": "distributed", "membership_id": "mem_456"}


class TestAdminRecoveryKeyResponse:
    def test_valid(self):
        resp = AdminRecoveryKeyResponse(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            admin_iv="cc" * 16,
            admin_tag="dd" * 16,
            admin_salt="ee" * 32,
        )
        assert resp.org_public_key == "aa" * 32
        assert resp.admin_encrypted_private_key == "bb" * 48
        assert resp.admin_iv == "cc" * 16
        assert resp.admin_tag == "dd" * 16
        assert resp.admin_salt == "ee" * 32

    def test_serialization(self):
        resp = AdminRecoveryKeyResponse(
            org_public_key="aa" * 32,
            admin_encrypted_private_key="bb" * 48,
            admin_iv="cc" * 16,
            admin_tag="dd" * 16,
            admin_salt="ee" * 32,
        )
        data = resp.model_dump()
        assert data == {
            "org_public_key": "aa" * 32,
            "admin_encrypted_private_key": "bb" * 48,
            "admin_iv": "cc" * 16,
            "admin_tag": "dd" * 16,
            "admin_salt": "ee" * 32,
        }


class TestAgentStateResponse:
    def test_with_state(self):
        resp = AgentStateResponse(
            agent_name="test",
            encryption_mode="zero_trust",
            has_state=True,
            encrypted_tarball="ff" * 100,
        )
        assert resp.agent_name == "test"
        assert resp.encryption_mode == "zero_trust"
        assert resp.has_state is True
        assert resp.encrypted_tarball == "ff" * 100

    def test_without_state(self):
        resp = AgentStateResponse(
            agent_name="new_agent",
            encryption_mode="background",
            has_state=False,
        )
        assert resp.has_state is False
        assert resp.encrypted_tarball is None

    def test_serialization(self):
        resp = AgentStateResponse(
            agent_name="test",
            encryption_mode="zero_trust",
            has_state=True,
        )
        data = resp.model_dump()
        assert data == {
            "agent_name": "test",
            "encryption_mode": "zero_trust",
            "has_state": True,
            "encrypted_tarball": None,
        }

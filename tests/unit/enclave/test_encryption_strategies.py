"""Unit tests for encryption strategy pattern."""

import pytest

from core.crypto import EncryptedPayload
from core.enclave.encryption_strategies import (
    BackgroundStrategy,
    EncryptionStrategy,
    ZeroTrustStrategy,
    get_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_encrypted_payload():
    """Create a sample EncryptedPayload with valid-length fields."""
    return EncryptedPayload(
        ephemeral_public_key=b"\xaa" * 32,
        iv=b"\xbb" * 16,
        ciphertext=b"\xcc" * 64,
        auth_tag=b"\xdd" * 16,
        hkdf_salt=b"\xee" * 32,
    )


@pytest.fixture
def sample_kms_envelope():
    """Create a sample KMS envelope with bytes values."""
    return {
        "encrypted_dek": b"\x01" * 32,
        "iv": b"\x02" * 16,
        "ciphertext": b"\x03" * 64,
        "auth_tag": b"\x04" * 16,
    }


@pytest.fixture
def sample_response_with_state():
    """Create a sample enclave response containing encrypted_state."""
    return {
        "encrypted_state": {
            "ephemeral_public_key": "aa" * 32,
            "iv": "bb" * 16,
            "ciphertext": "cc" * 64,
            "auth_tag": "dd" * 16,
            "hkdf_salt": "ee" * 32,
        },
        "other_field": "value",
    }


@pytest.fixture
def sample_response_without_state():
    """Create a sample enclave response with no encrypted_state."""
    return {"other_field": "value"}


# =============================================================================
# get_strategy()
# =============================================================================


class TestGetStrategy:
    """Tests for get_strategy() factory function."""

    def test_returns_zero_trust_for_zero_trust(self):
        strategy = get_strategy("zero_trust")
        assert isinstance(strategy, ZeroTrustStrategy)

    def test_returns_background_for_background(self):
        strategy = get_strategy("background")
        assert isinstance(strategy, BackgroundStrategy)

    def test_defaults_to_zero_trust_for_unknown(self):
        strategy = get_strategy("anything_else")
        assert isinstance(strategy, ZeroTrustStrategy)

    def test_defaults_to_zero_trust_for_empty_string(self):
        strategy = get_strategy("")
        assert isinstance(strategy, ZeroTrustStrategy)

    def test_all_strategies_implement_interface(self):
        for mode in ("zero_trust", "background"):
            strategy = get_strategy(mode)
            assert isinstance(strategy, EncryptionStrategy)


# =============================================================================
# ZeroTrustStrategy
# =============================================================================


class TestZeroTrustStrategy:
    """Tests for ZeroTrustStrategy."""

    def test_prepare_state_with_payload(self, sample_encrypted_payload):
        """Returns dict from EncryptedPayload.to_dict()."""
        strategy = ZeroTrustStrategy()
        result = strategy.prepare_state_for_vsock(
            encrypted_state=sample_encrypted_payload,
        )
        assert result is not None
        assert isinstance(result, dict)
        assert result["ephemeral_public_key"] == "aa" * 32
        assert result["iv"] == "bb" * 16
        assert result["ciphertext"] == "cc" * 64
        assert result["auth_tag"] == "dd" * 16
        assert result["hkdf_salt"] == "ee" * 32

    def test_prepare_state_without_payload(self):
        """Returns None when no encrypted state provided."""
        strategy = ZeroTrustStrategy()
        result = strategy.prepare_state_for_vsock()
        assert result is None

    def test_prepare_state_ignores_kms_envelope(self, sample_kms_envelope):
        """KMS envelope is ignored in zero trust mode."""
        strategy = ZeroTrustStrategy()
        result = strategy.prepare_state_for_vsock(kms_envelope=sample_kms_envelope)
        assert result is None

    def test_extract_state_with_encrypted_state(self, sample_response_with_state):
        """Extracts EncryptedPayload from response."""
        strategy = ZeroTrustStrategy()
        result = strategy.extract_state_from_response(sample_response_with_state)

        assert result["kms_envelope"] is None
        assert result["encrypted_state"] is not None
        assert isinstance(result["encrypted_state"], EncryptedPayload)
        assert result["encrypted_state"].ephemeral_public_key == b"\xaa" * 32

    def test_extract_state_without_encrypted_state(self, sample_response_without_state):
        """Returns None encrypted_state when response has none."""
        strategy = ZeroTrustStrategy()
        result = strategy.extract_state_from_response(sample_response_without_state)

        assert result["encrypted_state"] is None
        assert result["kms_envelope"] is None

    def test_extract_state_empty_encrypted_state(self):
        """Returns None when encrypted_state is empty/falsy."""
        strategy = ZeroTrustStrategy()
        result = strategy.extract_state_from_response({"encrypted_state": {}})
        assert result["encrypted_state"] is None

        result = strategy.extract_state_from_response({"encrypted_state": None})
        assert result["encrypted_state"] is None


# =============================================================================
# BackgroundStrategy
# =============================================================================


class TestBackgroundStrategy:
    """Tests for BackgroundStrategy."""

    def test_prepare_state_with_kms_envelope(self, sample_kms_envelope):
        """Returns hex-encoded KMS envelope dict."""
        strategy = BackgroundStrategy()
        result = strategy.prepare_state_for_vsock(kms_envelope=sample_kms_envelope)

        assert result is not None
        assert isinstance(result, dict)
        assert result["encrypted_dek"] == "01" * 32
        assert result["iv"] == "02" * 16
        assert result["ciphertext"] == "03" * 64
        assert result["auth_tag"] == "04" * 16

    def test_prepare_state_without_kms_envelope(self):
        """Returns None when no KMS envelope provided."""
        strategy = BackgroundStrategy()
        result = strategy.prepare_state_for_vsock()
        assert result is None

    def test_prepare_state_ignores_encrypted_state(self, sample_encrypted_payload):
        """EncryptedPayload is ignored in background mode."""
        strategy = BackgroundStrategy()
        result = strategy.prepare_state_for_vsock(
            encrypted_state=sample_encrypted_payload,
        )
        assert result is None

    def test_extract_state_with_kms_envelope(self):
        """Extracts raw KMS envelope dict from response."""
        kms_data = {
            "encrypted_dek": "ab" * 32,
            "iv": "cd" * 16,
            "ciphertext": "ef" * 64,
            "auth_tag": "12" * 16,
        }
        strategy = BackgroundStrategy()
        result = strategy.extract_state_from_response({"encrypted_state": kms_data})

        assert result["encrypted_state"] is None
        assert result["kms_envelope"] is not None
        assert result["kms_envelope"] == kms_data

    def test_extract_state_without_encrypted_state(self, sample_response_without_state):
        """Returns None kms_envelope when response has none."""
        strategy = BackgroundStrategy()
        result = strategy.extract_state_from_response(sample_response_without_state)

        assert result["encrypted_state"] is None
        assert result["kms_envelope"] is None

    def test_extract_state_empty_encrypted_state(self):
        """Returns None when encrypted_state is empty/falsy."""
        strategy = BackgroundStrategy()
        result = strategy.extract_state_from_response({"encrypted_state": {}})
        assert result["kms_envelope"] is None

        result = strategy.extract_state_from_response({"encrypted_state": None})
        assert result["kms_envelope"] is None

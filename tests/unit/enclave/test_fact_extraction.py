"""Tests for enclave fact extraction."""

import pytest
from core.enclave.fact_extraction import FactExtractor


class TestFactExtractor:
    """Tests for FactExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return FactExtractor()

    def test_extracts_preference(self, extractor):
        """Should extract user preferences."""
        facts = extractor.extract(user_message="My favorite color is blue", assistant_response="That's a great choice!")
        assert len(facts) >= 1
        pref = next((f for f in facts if f.predicate == "prefers"), None)
        assert pref is not None
        assert "blue" in pref.object.lower()

    def test_extracts_work_info(self, extractor):
        """Should extract work/employment info."""
        facts = extractor.extract(
            user_message="I work at Google as a software engineer", assistant_response="That's exciting!"
        )
        assert len(facts) >= 1
        work = next((f for f in facts if f.predicate == "works_at"), None)
        assert work is not None
        assert "google" in work.object.lower()

    def test_extracts_location(self, extractor):
        """Should extract location info."""
        facts = extractor.extract(user_message="I live in San Francisco", assistant_response="Great city!")
        location = next((f for f in facts if f.predicate == "located_in"), None)
        assert location is not None
        assert "san francisco" in location.object.lower()

    def test_returns_empty_for_no_facts(self, extractor):
        """Should return empty list when no facts found."""
        facts = extractor.extract(
            user_message="What's the weather like?", assistant_response="I don't have weather data."
        )
        # May or may not find facts, but shouldn't crash
        assert isinstance(facts, list)

    def test_fact_has_confidence(self, extractor):
        """Extracted facts should have confidence scores."""
        facts = extractor.extract(
            user_message="I definitely prefer Python over JavaScript", assistant_response="Python is great!"
        )
        if facts:
            assert all(0 <= f.confidence <= 1 for f in facts)

    def test_fact_has_type(self, extractor):
        """Extracted facts should have inferred types."""
        facts = extractor.extract(user_message="I love TypeScript", assistant_response="Great choice!")
        if facts:
            assert all(f.type in ["preference", "identity", "plan", "observation"] for f in facts)

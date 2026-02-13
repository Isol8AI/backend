"""Tests for TownSimulation engine."""

import pytest

from core.services.town_simulation import TownSimulation, TOWN_LOCATIONS


class TestTownLocations:
    """Test town location definitions."""

    def test_locations_defined(self):
        assert "home" in TOWN_LOCATIONS
        assert "cafe" in TOWN_LOCATIONS
        assert "plaza" in TOWN_LOCATIONS
        assert "library" in TOWN_LOCATIONS

    def test_locations_have_coordinates(self):
        for name, loc in TOWN_LOCATIONS.items():
            assert "x" in loc
            assert "y" in loc
            assert isinstance(loc["x"], (int, float))
            assert isinstance(loc["y"], (int, float))


class TestTownSimulation:
    """Test simulation tick logic."""

    def test_pick_random_location(self):
        sim = TownSimulation.__new__(TownSimulation)
        current = "home"
        target = sim._pick_random_location(exclude=current)
        assert target != current
        assert target in TOWN_LOCATIONS

    def test_calculate_distance(self):
        sim = TownSimulation.__new__(TownSimulation)
        dist = sim._calculate_distance(0, 0, 3, 4)
        assert dist == 5.0

    def test_move_toward_target(self):
        sim = TownSimulation.__new__(TownSimulation)
        new_x, new_y, arrived = sim._move_toward(
            current_x=0.0,
            current_y=0.0,
            target_x=100.0,
            target_y=0.0,
            speed=10.0,
        )
        assert new_x == 10.0
        assert new_y == 0.0
        assert arrived is False

    def test_move_toward_arrives_when_close(self):
        sim = TownSimulation.__new__(TownSimulation)
        new_x, new_y, arrived = sim._move_toward(
            current_x=95.0,
            current_y=0.0,
            target_x=100.0,
            target_y=0.0,
            speed=10.0,
        )
        assert new_x == 100.0
        assert new_y == 0.0
        assert arrived is True

    def test_should_converse_probability(self):
        sim = TownSimulation.__new__(TownSimulation)
        # Strangers: 15% base probability
        assert sim._conversation_probability(0) == 0.15
        # Friends (50+ affinity): higher probability
        assert sim._conversation_probability(50) > 0.15
        # Close friends (80+ affinity): even higher
        assert sim._conversation_probability(80) > sim._conversation_probability(50)

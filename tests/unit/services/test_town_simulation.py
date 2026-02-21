"""Tests for TownSimulation engine."""

from core.town_constants import TOWN_LOCATIONS
from core.services.town_simulation import TownSimulation


class TestTownLocations:
    """Test town location definitions."""

    def test_locations_defined(self):
        assert "home" in TOWN_LOCATIONS
        assert "cafe" in TOWN_LOCATIONS
        assert "plaza" in TOWN_LOCATIONS
        assert "library" in TOWN_LOCATIONS
        assert "park" in TOWN_LOCATIONS
        assert "shop" in TOWN_LOCATIONS

    def test_locations_have_coordinates(self):
        for name, loc in TOWN_LOCATIONS.items():
            assert "x" in loc
            assert "y" in loc
            assert isinstance(loc["x"], (int, float))
            assert isinstance(loc["y"], (int, float))


class TestTownSimulationConstants:
    """Test simulation timing and speed constants."""

    def test_tick_interval_is_fast_enough(self):
        from core.services.town_simulation import TICK_INTERVAL
        assert TICK_INTERVAL <= 3.0, "Tick should be <=3s for smooth animation"

    def test_agent_speed_reasonable(self):
        from core.services.town_simulation import AGENT_SPEED
        assert 0.1 <= AGENT_SPEED <= 2.0, "Speed should be in tile/tick range"

    def test_decision_cooldown_reasonable(self):
        from core.services.town_simulation import DECISION_COOLDOWN
        assert DECISION_COOLDOWN >= 5.0, "Agents shouldn't decide too fast"


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

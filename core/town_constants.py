"""Shared constants for the GooseTown simulation.

Extracted here to avoid circular imports between routers/town.py and
core/services/town_simulation.py.
"""

from typing import Dict, List

# Town locations in tile coordinates (verified walkable on gentle_map.json).
# Each location has 4+ walkable neighbours so agents can approach from any side.
TOWN_LOCATIONS: Dict[str, Dict] = {
    "cafe": {"x": 6.0, "y": 6.0, "label": "Cafe"},
    "plaza": {"x": 12.0, "y": 20.0, "label": "Town Plaza"},
    "library": {"x": 31.0, "y": 7.0, "label": "Library"},
    "park": {"x": 32.0, "y": 21.0, "label": "Park"},
    "shop": {"x": 17.0, "y": 10.0, "label": "General Store"},
    "home": {"x": 10.0, "y": 14.0, "label": "Home"},
}

# Default characters for the town (same data used by both simulation and router)
DEFAULT_CHARACTERS: List[Dict] = [
    {
        "name": "Lucky",
        "agent_name": "lucky",
        "character": "f1",
        "identity": (
            "Lucky is always happy and curious, and he loves cheese. He spends "
            "most of his time reading about the history of science and traveling "
            "through the galaxy on whatever ship will take him."
        ),
        "plan": "You want to hear all the gossip.",
        "spawn": {"x": 8.0, "y": 6.0},
        "home": "cafe",
    },
    {
        "name": "Bob",
        "agent_name": "bob",
        "character": "f4",
        "identity": (
            "Bob is always grumpy and he loves trees. He spends most of his time "
            "gardening by himself. When spoken to he'll respond but try and get "
            "out of the conversation as quickly as possible."
        ),
        "plan": "You want to avoid people as much as possible.",
        "spawn": {"x": 18.0, "y": 5.0},
        "home": "shop",
    },
    {
        "name": "Stella",
        "agent_name": "stella",
        "character": "f6",
        "identity": (
            "Stella can never be trusted. She tries to trick people all the time. "
            "She's incredibly charming and not afraid to use her charm."
        ),
        "plan": "You want to take advantage of others as much as possible.",
        "spawn": {"x": 12.0, "y": 13.0},
        "home": "home",
    },
    {
        "name": "Alice",
        "agent_name": "alice",
        "character": "f3",
        "identity": (
            "Alice is a famous scientist. She is smarter than everyone else and "
            "has discovered mysteries of the universe no one else can understand."
        ),
        "plan": "You want to figure out how the world works.",
        "spawn": {"x": 31.0, "y": 5.0},
        "home": "library",
    },
    {
        "name": "Pete",
        "agent_name": "pete",
        "character": "f7",
        "identity": (
            "Pete is deeply religious and sees the hand of god or of the work of "
            "the devil everywhere. He can't have a conversation without bringing "
            "up his deep faith."
        ),
        "plan": "You want to convert everyone to your religion.",
        "spawn": {"x": 10.0, "y": 17.0},
        "home": "plaza",
    },
]

# Default spawn positions (derived from DEFAULT_CHARACTERS for backward compat)
DEFAULT_SPAWN_POSITIONS = [c["spawn"] for c in DEFAULT_CHARACTERS]

# Available character sprites (f1-f8 in data/characters.ts)
AVAILABLE_CHARACTERS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]

# Characters already assigned to default AI agents
AGENT_CHARACTERS = {c["character"] for c in DEFAULT_CHARACTERS}

# System user ID for default seeded agents
SYSTEM_USER_ID = "system"

# Walk speed reported to the frontend for interpolation (tiles/tick at 2s ticks)
WALK_SPEED_DISPLAY = 0.6

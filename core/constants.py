"""
Application-wide constants.

These constants are used across the codebase for consistency and maintainability.
"""

# System actor ID used for audit logs when action is performed by the system
# (e.g., Clerk webhook processing) rather than a human user.
# This provides a consistent identifier for system-initiated actions.
SYSTEM_ACTOR_ID = "__system__"

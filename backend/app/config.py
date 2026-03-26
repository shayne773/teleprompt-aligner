"""Application configuration and lightweight demo data.

This module centralizes values that are likely to be reused across routes
and services while the project is still in early development.
"""

from dataclasses import dataclass, field


# Core constants
APP_NAME = "Teleprompt Aligner API"
WEBSOCKET_ROUTE = "/ws"


# Tiny demo script payload for local testing before real script loading exists.
DEMO_SCRIPT_LINES = [
	"Welcome to Teleprompt Aligner.",
	"This is a demo line for socket testing.",
	"When you are ready, connect real script loading.",
]


@dataclass(frozen=True)
class Settings:
	"""Simple settings container for app-wide configuration values."""

	app_name: str = APP_NAME
	websocket_route: str = WEBSOCKET_ROUTE
	demo_script_lines: list[str] = field(default_factory=lambda: list(DEMO_SCRIPT_LINES))


settings = Settings()
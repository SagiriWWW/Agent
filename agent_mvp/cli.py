"""Compatibility shim -> aircraft_agent.interface.cli."""
from aircraft_agent.interface.cli import *  # noqa: F401,F403
from aircraft_agent.interface.cli import main  # noqa: F401

if __name__ == "__main__":
    main()

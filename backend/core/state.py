import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerState:
    ready: bool = False
    started_at: float = field(default_factory=time.time)
    init_error: Optional[str] = None
    checks_passed: list = field(default_factory=list)


# Global singleton — imported by lifespan, middleware, and health routes
server_state = ServerState()

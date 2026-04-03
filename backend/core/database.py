from typing import Dict, Any

# In-memory store: startup_id -> { status, result, pipeline_status, ... }
db: Dict[str, Dict[str, Any]] = {}

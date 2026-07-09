# Datamining FP-growth Backend

This vendored directory is a lightweight FP-growth-only backend for the Agent
MVP. It preserves the integration surface needed by `agent_mvp` while excluding
other association-mining implementations and comparison scripts.

Required exported symbols:

- `fpgrowth`
- `association_rules`
- `is_same_sensor_rule`

The Agent adapter loads `scripts/core/fpgrowth_core.py` dynamically.


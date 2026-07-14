# Public JSON Schemas

This directory contains committed snapshots of TinyLLM-System's versioned public schemas.
They are generated from strict Pydantic models and checked in CI.

Regenerate snapshots after an intentional, versioned contract change:

```bash
.venv/bin/python scripts/export_schemas.py
```

Never edit a generated schema without changing its source model. Breaking changes require a new
schema version and migration tests.

"""Benchmark data and the harness that reads it.

The command set, the paraphrase groups and the spec-to-production name mapping
live here as **data files**, not as literals inside a screen or a test. A
benchmark hardcoded into the interface cannot be re-run headlessly, cannot be
diffed when it changes, and quietly becomes whatever the interface needed.

Nothing in this package is imported by the runtime pipeline.
"""

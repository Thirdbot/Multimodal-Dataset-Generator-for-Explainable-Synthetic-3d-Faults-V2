"""Trace one (or more) finished builds into a property graph, in an isolated subprocess.

Invoked by scripts/watcher/process.py `_run_trace` (via `python -m scripts.graph.trace_one <build>...`)
so the GIL-bound trace work parallelizes across cores (TRACE_CONCURRENCY subprocesses) instead of
serializing in the watcher's asyncio thread pool.
"""
import sys
from pathlib import Path

from scripts.graph.graph_generator import trace_success_tracker

if __name__ == "__main__":
    trace_success_tracker([Path(p) for p in sys.argv[1:]])

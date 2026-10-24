"""The orchestrator that wires every other package together.

``core`` owns the pipeline described in specification section 9 -- text in,
intent, plan, permission check, risk validation, registry lookup, execution,
response out -- and nothing else.  It is deliberately separate from ``ui`` so
the text interface, the dashboard and the future voice loop all drive the
*same* engine rather than each reimplementing the security checks.
"""

from core.connectivity import ConnectivityMonitor, NetworkMode
from core.engine import AarmaEngine, EngineResponse, ResponseKind

__all__ = [
    "AarmaEngine",
    "ConnectivityMonitor",
    "EngineResponse",
    "NetworkMode",
    "ResponseKind",
]

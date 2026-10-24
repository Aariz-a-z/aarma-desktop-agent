"""Cross-cutting helpers shared by every AARMA subsystem.

Nothing in this package may import from :mod:`ai`, :mod:`actions`, :mod:`ui`
or :mod:`security`.  Utilities sit at the bottom of the dependency graph so
that they can be used from anywhere without creating import cycles.
"""

from utils.paths import AppPaths, get_paths

__all__ = ["AppPaths", "get_paths"]

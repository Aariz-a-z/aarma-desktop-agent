"""Discovering what is actually on this computer, inside approved boundaries.

Phase 5 widens AARMA from a fixed catalogue to "find the real target".  That is
three services and one policy, and the policy comes first in every path:

* :mod:`workspace.policy` -- where AARMA may work, and what it may do there.
  Independent of the language layer by design: the resolver may propose any
  target, and this decides whether it is allowed.
* :mod:`workspace.repositories` -- the user's projects, found by their markers
  rather than from a list somebody had to write.
* :mod:`workspace.applications` -- what is installed, from Start Menu shortcuts
  and PATH rather than by crawling the disk.

Nothing here executes anything.  Discovery answers "what is there"; the action
layer decides what to do about it, and the safety layer decides whether that is
allowed.  Keeping the three apart is what stops a broader search from becoming
a broader set of powers.
"""

from workspace.policy import (
    AccessDecision,
    Operation,
    Verdict,
    WorkspaceAccessPolicy,
)
from workspace.repositories import Repository, RepositoryIndex

__all__ = ["AccessDecision", "Operation", "Repository", "RepositoryIndex",
           "Verdict", "WorkspaceAccessPolicy"]

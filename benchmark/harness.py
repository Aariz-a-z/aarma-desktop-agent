"""Loading the benchmark, and running it against the intent engine as text.

Two things live here, deliberately separated:

:class:`BenchmarkSuite` reads the data files and turns spec names into
production names. The translation is **expectation-side only** -- it builds what
we expect to see, and nothing in the runtime pipeline ever reads it. Production
is not renamed to fit the benchmark, because then the benchmark would stop
measuring production.

:func:`run_text_only` feeds command strings straight into
:class:`~ai.intent.IntentEngine`, skipping audio entirely. That separation is
the point of the exercise: if intent accuracy is already poor on perfect text,
the speech layer is not the primary problem, and no amount of work on the
microphone will fix it.

A command whose spec intent has no production equivalent is marked
``NO_PRODUCTION_INTENT`` and left out of the denominator. Scoring it as a failure
would blame the engine for a capability nobody has built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DATA_DIR = Path(__file__).resolve().parent

NO_PRODUCTION_INTENT = "NO_PRODUCTION_INTENT"

#: Intents the planner answers itself instead of dispatching to the registry.
#: Kept here so the harness can tell "no action, by design" from "no action,
#: because something broke".
META_INTENTS = frozenset({"exit", "help", "greeting", "thanks",
                          "confirm", "decline", "unknown"})


def _load(name: str) -> Dict[str, Any]:
    with (DATA_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ExpectedCommand:
    """One benchmark line, with spec names already resolved to production."""

    id: int
    text: str
    spec_intent: str
    spec_parameters: Dict[str, Any]
    intent: Optional[str]                  # None when unmapped.
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolvable(self) -> bool:
        """Whether production has any intent that could satisfy this line."""
        return self.intent is not None

    @property
    def mapping_note(self) -> str:
        """Shown on every row, so a mapping mistake is visible not silent."""
        if self.intent is None:
            return f"{self.spec_intent} -> {NO_PRODUCTION_INTENT}"
        if self.intent != self.spec_intent:
            return f"{self.spec_intent} -> {self.intent}"
        return self.intent


class BenchmarkSuite:
    """The command set, the paraphrase groups, and the name mapping."""

    def __init__(self, *, commands: Optional[Dict[str, Any]] = None,
                 mapping: Optional[Dict[str, Any]] = None,
                 paraphrases: Optional[Dict[str, Any]] = None) -> None:
        self._commands = commands or _load("commands.json")
        self._mapping = mapping or _load("intent_mapping.json")
        self._paraphrases = paraphrases or _load("paraphrases.json")

        self._intent_map: Dict[str, str] = self._mapping.get("intents", {})
        self._parameter_map: Dict[str, str] = self._mapping.get("parameters", {})
        self._value_map: Dict[str, Any] = {
            key: value for key, value in
            self._mapping.get("parameter_values", {}).items()
            if key != "description"
        }
        self._known: Optional[frozenset] = None

    # ------------------------------------------------------------- mapping

    def production_intent(self, spec_intent: str) -> str:
        return self._intent_map.get(spec_intent, spec_intent)

    def production_parameters(self, spec_intent: str,
                              spec_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rename parameter keys, and translate values where production differs.

        The value translation exists for one measured reason: ``open_url`` takes
        a bare shortcut token (``youtube``), not a domain and not a full URL.
        That was read out of ``actions/browser.py`` and ``ai/intent.py`` rather
        than guessed, and it is recorded in the mapping file so the expectation
        can be checked against the code.
        """
        intent = self.production_intent(spec_intent)
        values = self._value_map.get(intent, {})
        renamed: Dict[str, Any] = {}
        for key, value in (spec_parameters or {}).items():
            production_key = self._parameter_map.get(key, key)
            table = values.get(production_key, {})
            if isinstance(value, str) and isinstance(table, dict):
                value = table.get(value, value)
            renamed[production_key] = value
        return renamed

    def known_intents(self) -> frozenset:
        """Every intent production could actually produce: registry + meta."""
        if self._known is None:
            from actions.registry import build_default_registry

            registry = build_default_registry()
            self._known = frozenset({spec.name for spec in registry.all()}
                                    | META_INTENTS)
        return self._known

    def resolve(self, spec_intent: str) -> Optional[str]:
        """The production intent, or ``None`` if nothing can satisfy it."""
        candidate = self.production_intent(spec_intent)
        return candidate if candidate in self.known_intents() else None

    # -------------------------------------------------------------- loading

    def commands(self) -> List[ExpectedCommand]:
        resolved: List[ExpectedCommand] = []
        for entry in self._commands["commands"]:
            spec_intent = entry["intent"]
            spec_parameters = entry.get("parameters", {}) or {}
            production = self.resolve(spec_intent)
            resolved.append(ExpectedCommand(
                id=entry["id"], text=entry["text"],
                spec_intent=spec_intent, spec_parameters=spec_parameters,
                intent=production,
                parameters=(self.production_parameters(spec_intent, spec_parameters)
                            if production else {}),
            ))
        return resolved

    def paraphrase_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for group in self._paraphrases["groups"]:
            spec_intent = group["intent"]
            groups.append({
                "name": group["name"],
                "spec_intent": spec_intent,
                "intent": self.resolve(spec_intent),
                "parameters": self.production_parameters(
                    spec_intent, group.get("parameters", {}) or {}),
                "inputs": list(group["inputs"]),
            })
        return groups

    def short_controls(self) -> List[Dict[str, Any]]:
        controls: List[Dict[str, Any]] = []
        for entry in self._paraphrases.get("short_controls", []):
            spec_intent = entry.get("intent")
            controls.append({
                "input": entry["input"],
                "spec_intent": spec_intent,
                "intent": self.resolve(spec_intent) if spec_intent else None,
                "parameters": (self.production_parameters(
                    spec_intent, entry.get("parameters", {}) or {})
                    if spec_intent else {}),
                "note": entry.get("note", ""),
            })
        return controls


# ------------------------------------------------------------- text harness

@dataclass
class TextResult:
    """One command run through the intent engine as text."""

    text: str
    normalized: str
    expected_intent: Optional[str]
    expected_parameters: Dict[str, Any]
    matched_intent: str
    matched_parameters: Dict[str, Any]
    match_method: str
    match_score: float
    excluded: bool = False          # NO_PRODUCTION_INTENT.
    mapping_note: str = ""
    command_id: Optional[int] = None

    @property
    def intent_ok(self) -> bool:
        return (self.expected_intent is not None
                and self.matched_intent == self.expected_intent)

    @property
    def parameters_ok(self) -> bool:
        """Only the expected keys are checked; extras are not failures."""
        for key, want in (self.expected_parameters or {}).items():
            if key not in self.matched_parameters:
                return False
            got = self.matched_parameters[key]
            if isinstance(want, str) and isinstance(got, str):
                if want.strip().casefold() != got.strip().casefold():
                    return False
            elif want != got:
                return False
        return True

    @property
    def passed(self) -> bool:
        return (not self.excluded) and self.intent_ok and self.parameters_ok

    @property
    def verdict(self) -> str:
        if self.excluded:
            return NO_PRODUCTION_INTENT
        if self.passed:
            return "PASS"
        if self.matched_intent == "unknown":
            return "FAIL (no match)"
        if not self.intent_ok:
            return "FAIL (wrong intent)"
        return "FAIL (wrong parameters)"


def _run_one(engine, text: str, expected_intent: Optional[str],
             expected_parameters: Dict[str, Any], *, excluded: bool,
             mapping_note: str = "", command_id: Optional[int] = None) -> TextResult:
    intent = engine.parse(text)
    return TextResult(
        text=text,
        normalized=intent.normalised_text,
        expected_intent=expected_intent,
        expected_parameters=expected_parameters,
        matched_intent=intent.name,
        matched_parameters=dict(intent.parameters),
        match_method=intent.match_method,
        match_score=intent.confidence,
        excluded=excluded,
        mapping_note=mapping_note,
        command_id=command_id,
    )


def run_text_only(engine, suite: Optional[BenchmarkSuite] = None
                  ) -> List[TextResult]:
    """The 20 commands, fed straight in as text. No microphone involved."""
    active = suite or BenchmarkSuite()
    return [
        _run_one(engine, command.text, command.intent, command.parameters,
                 excluded=not command.resolvable,
                 mapping_note=command.mapping_note, command_id=command.id)
        for command in active.commands()
    ]


def run_paraphrases(engine, suite: Optional[BenchmarkSuite] = None
                    ) -> Dict[str, List[TextResult]]:
    """Every paraphrase group, keyed by group name."""
    active = suite or BenchmarkSuite()
    results: Dict[str, List[TextResult]] = {}
    for group in active.paraphrase_groups():
        results[group["name"]] = [
            _run_one(engine, text, group["intent"], group["parameters"],
                     excluded=group["intent"] is None,
                     mapping_note=f"{group['spec_intent']} -> "
                                  f"{group['intent'] or NO_PRODUCTION_INTENT}")
            for text in group["inputs"]
        ]
    return results


def run_short_controls(engine, suite: Optional[BenchmarkSuite] = None
                       ) -> List[TextResult]:
    """One- and two-word utterances, including ones with no production intent."""
    active = suite or BenchmarkSuite()
    return [
        _run_one(engine, control["input"], control["intent"],
                 control["parameters"],
                 excluded=control["intent"] is None,
                 mapping_note=control["note"] or (control["intent"] or ""))
        for control in active.short_controls()
    ]


def score(results: Sequence[TextResult]) -> Dict[str, Any]:
    """Counts, with the denominator kept visible."""
    scored = [r for r in results if not r.excluded]
    excluded = [r for r in results if r.excluded]
    passes = sum(1 for r in scored if r.passed)
    return {
        "passed": passes,
        "total": len(scored),
        "excluded": len(excluded),
        "accuracy": (passes / len(scored)) if scored else 0.0,
        "failures": [r for r in scored if not r.passed],
        "excluded_rows": excluded,
    }

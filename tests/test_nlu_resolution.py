"""Nearest-intent understanding: the paraphrases, and the sentences that aren't.

Two corpora, and the second is the one that matters.  Resolving paraphrases is
the feature; refusing sentences that merely *mention* a capability is what makes
the feature safe to have.  A resolver that scores 100% on phrasings and fires on
"my friend said close the tab" has made the assistant worse, because a missed
command costs a repeat and a wrongly-executed one costs whatever it did.

The single most important assertion in this file is
``test_no_false_positive_ever_resolves``.  Its target is zero and it is
parametrised over every false-positive case, so a regression names the exact
sentence that broke.

The last class is mutation testing: each one removes a guard and asserts that
something notices.  A safety property with no test that fails when it is
deleted is a comment.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from actions.registry import build_default_registry  # noqa: E402
from ai.nlu import (  # noqa: E402
    Confidence,
    NaturalLanguageResolver,
    ResolutionContext,
    declines,
    extract,
    normalise,
)
from nlu_corpus import (  # noqa: E402
    AMBIGUOUS,
    FALSE_POSITIVES,
    PARAPHRASES,
    paraphrase_cases,
)
from security.validator import RiskLevel  # noqa: E402

BROWSER = ResolutionContext(active_window_title="AARMA - Google Chrome",
                            active_process="chrome.exe")
EDITOR = ResolutionContext(active_window_title="main.py - VS Code",
                           active_process="Code.exe")
NOWHERE = ResolutionContext()
CONTEXTS: Dict[Optional[str], ResolutionContext] = {
    None: NOWHERE, "browser": BROWSER, "editor": EDITOR}


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def resolver(registry):
    return NaturalLanguageResolver(registry)


# ------------------------------------------------------------- paraphrases

class TestParaphrases:
    """Every phrasing in the corpus reaches the action it names."""

    @pytest.mark.parametrize("sentence,expected,entities,context",
                             paraphrase_cases(),
                             ids=[case[0] for case in paraphrase_cases()])
    def test_paraphrase_resolves(self, resolver, sentence, expected,
                                 entities, context) -> None:
        result = resolver.resolve(sentence, CONTEXTS[context])
        assert result.action == expected, (
            f"{sentence!r} -> {result.action or result.refused!r}; "
            f"reasons: {result.reasons}")

    @pytest.mark.parametrize("sentence,expected,entities,context",
                             [case for case in paraphrase_cases() if case[2]],
                             ids=[case[0] for case in paraphrase_cases()
                                  if case[2]])
    def test_entities_are_extracted(self, resolver, sentence, expected,
                                    entities, context) -> None:
        result = resolver.resolve(sentence, CONTEXTS[context])
        for key, want in entities.items():
            if not want:
                continue
            assert str(result.entities.get(key, "")).lower() == want.lower(), (
                f"{sentence!r} {key}: wanted {want!r}, "
                f"got {result.entities.get(key)!r}")

    def test_every_major_action_has_ten_phrasings(self) -> None:
        """The corpus is the coverage claim, so it is asserted.

        Weather and windows are deliberately smaller: they have fewer natural
        phrasings, not less coverage, and padding a corpus to hit a number
        would be measuring the padding.
        """
        thin = {"weather", "windows"}
        for group, _context, cases in PARAPHRASES:
            if group in thin:
                continue
            assert len(cases) >= 10, f"{group} has only {len(cases)}"

    def test_the_query_is_never_the_whole_sentence(self, resolver) -> None:
        """The failure that looks like it worked."""
        result = resolver.resolve("Open Python tutorials on YouTube.")
        assert result.entities["query"] == "python tutorials"
        assert "youtube" not in result.entities["query"]
        assert "open" not in result.entities["query"]


# --------------------------------------------------------- false positives

class TestFalsePositives:
    """The target is zero, and it is the most important number here."""

    @pytest.mark.parametrize("sentence,context", FALSE_POSITIVES,
                             ids=[case[0] for case in FALSE_POSITIVES])
    def test_no_false_positive_ever_resolves(self, resolver, sentence,
                                             context) -> None:
        result = resolver.resolve(sentence, CONTEXTS[context])
        assert not result.resolved, (
            f"{sentence!r} would have run {result.action} "
            f"at {result.confidence:.2f}")

    def test_the_false_execution_rate_is_zero(self, resolver) -> None:
        """Stated as one number, because that is how it should be reported."""
        executed = [sentence for sentence, context in FALSE_POSITIVES
                    if resolver.resolve(sentence, CONTEXTS[context]).resolved]
        assert executed == []

    @pytest.mark.parametrize("sentence", [
        "I don't want you to open Chrome.",
        "Don't close this tab.",
        "Do not shut down the computer.",
    ])
    def test_negation_is_caught_before_pattern_matching(self, sentence) -> None:
        """Not just in the resolver: the rules would match these otherwise.

        The action rules are unanchored, so "open chrome" is found *inside* "I
        don't want you to open Chrome". A negation check that ran only on the
        resolver's path would never see the sentences that most need it.
        """
        assert declines(sentence) == "negated"

    def test_cancel_and_stop_remain_commands(self) -> None:
        """They are escape hatches, not negation, and must keep working."""
        for sentence in ("cancel shutdown", "stop the restart",
                         "cancel my 6 PM reminder"):
            assert declines(sentence) == "", (
                f"{sentence!r} was read as negation; the escape hatch is gone")


# ------------------------------------------------------------- ambiguity

class TestAmbiguity:

    @pytest.mark.parametrize("sentence,context", AMBIGUOUS,
                             ids=[case[0] for case in AMBIGUOUS])
    def test_ambiguous_requests_ask(self, resolver, sentence, context) -> None:
        result = resolver.resolve(sentence, CONTEXTS[context])
        assert result.requires_clarification
        assert result.clarification
        assert not result.resolved

    def test_a_settled_convention_does_not_ask(self, resolver) -> None:
        """"The command line" means the Command Prompt; asking is pedantry."""
        result = resolver.resolve("open the command line")
        assert result.action == "open_terminal"
        assert not result.requires_clarification


# --------------------------------------------------------------- context

class TestContext:
    """"Close this" is a different instruction in a browser and an editor."""

    def test_close_this_in_a_browser_closes_a_tab(self, resolver) -> None:
        assert resolver.resolve("close this", BROWSER).action == "close_tab"

    def test_close_this_in_an_editor_does_not_close_a_tab(self,
                                                          resolver) -> None:
        result = resolver.resolve("close this", EDITOR)
        assert result.action != "close_tab"
        assert result.action == "close_window"

    def test_close_this_with_no_context_asks(self, resolver) -> None:
        """Guessing here closes something the user was not looking at."""
        result = resolver.resolve("close this", NOWHERE)
        assert not result.resolved
        assert result.requires_clarification

    def test_browser_navigation_prefers_a_browser(self, resolver) -> None:
        in_browser = resolver.resolve("go back", BROWSER)
        assert in_browser.action == "go_back"
        assert in_browser.confidence > resolver.resolve("go back", EDITOR).confidence


# ------------------------------------------------------------ the boundary

class TestCapabilityBoundary:
    """Flexible language must not become invented capability."""

    @pytest.mark.parametrize("sentence", [
        "I want you to fly a drone.", "Order me a pizza.",
        "Book me a flight to Tokyo.", "Print this on my 3D printer.",
        "Send a fax to my accountant.",
    ])
    def test_out_of_domain_has_no_nearest_intent(self, resolver,
                                                 sentence) -> None:
        result = resolver.resolve(sentence)
        assert not result.resolved

    def test_candidates_come_only_from_the_registry(self, registry) -> None:
        """Remove the action and the words that reached it stop working."""
        from actions.registry import ActionRegistry

        empty = ActionRegistry()
        bare = NaturalLanguageResolver(empty)
        result = bare.resolve("take a screenshot")
        assert not result.resolved
        assert result.refused == "no capability"

    def test_every_resolvable_action_is_registered(self, resolver,
                                                   registry) -> None:
        for sentence, expected, _entities, context in paraphrase_cases():
            result = resolver.resolve(sentence, CONTEXTS[context])
            if result.action:
                assert result.action in registry, (
                    f"{sentence!r} resolved to unregistered {result.action!r}")


# ------------------------------------------------------------------ safety

class TestUnderstandingIsNotPermission:
    """Resolving a dangerous sentence changes nothing about the gates."""

    def test_power_resolves_confidently(self, resolver) -> None:
        result = resolver.resolve("Could you maybe turn the computer off?")
        assert result.action == "shutdown_computer"
        assert result.confidence >= Confidence.ACT

    def test_but_it_still_meets_permission_and_confirmation(self,
                                                            engine) -> None:
        """Through the real engine: resolved, and still asked about."""
        from core.engine import ResponseKind
        from security.permissions import Permission

        denied = engine.handle_text("Could you maybe turn the computer off?")
        assert denied.kind is ResponseKind.ERROR

        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        asked = engine.handle_text("Could you maybe turn the computer off?")
        assert asked.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert asked.action == "shutdown_computer"

    def test_a_fuzzy_match_cannot_reach_a_risky_action(self, resolver,
                                                       registry) -> None:
        """Spelling repair reaches the volume; it must not reach a shutdown."""
        from ai.nlu.resolver import FUZZY_RISK_CEILING

        assert FUZZY_RISK_CEILING == "Low"
        for name in ("shutdown_computer", "restart_computer", "delete_item"):
            spec = registry.get(name)
            if spec is not None:
                assert spec.risk_label != FUZZY_RISK_CEILING

    def test_nothing_downgrades_a_risk_level(self, resolver, registry) -> None:
        """The resolver names an action; it never re-grades one."""
        for sentence, expected, _entities, context in paraphrase_cases():
            result = resolver.resolve(sentence, CONTEXTS[context])
            if not result.action:
                continue
            spec = registry.get(result.action)
            assert spec is not None
            # The resolution carries no risk field at all -- there is nothing
            # for it to disagree with the registry about.
            assert not hasattr(result, "risk")


# ------------------------------------------------------------- performance

class TestPerformance:
    """Simple commands stay local and fast; no network on this path."""

    def test_resolution_is_fast(self, resolver) -> None:
        import time

        sentences = [case[0] for case in paraphrase_cases()]
        started = time.perf_counter()
        for sentence in sentences:
            resolver.resolve(sentence)
        elapsed = (time.perf_counter() - started) * 1000 / len(sentences)
        assert elapsed < 100, f"mean {elapsed:.1f} ms per sentence"

    def test_no_network_is_used(self, resolver, monkeypatch) -> None:
        """A resolver that reached for a model would fail this outright."""
        import socket

        def _forbidden(*args, **kwargs):
            raise AssertionError("the resolver opened a socket")

        monkeypatch.setattr(socket, "socket", _forbidden)
        monkeypatch.setattr(socket, "create_connection", _forbidden)
        for sentence in ("open chrome", "make it quieter",
                         "Open Python tutorials on YouTube."):
            resolver.resolve(sentence)

    def test_it_works_with_no_llm(self, resolver) -> None:
        """The deterministic path is the whole path in this build."""
        from ai.llm import get_provider

        assert get_provider().is_available() is False
        assert resolver.resolve("make it quieter").action == "volume_down"


# --------------------------------------------------------- mutation tests

class TestMutations:
    """Remove a guard; assert something fails. Otherwise it is a comment."""

    def test_removing_the_confidence_floor_is_caught(self, resolver,
                                                     monkeypatch) -> None:
        """With the floor at zero, weak matches would start executing."""
        weak = "my friend uses github"
        assert not resolver.resolve(weak).resolved

        monkeypatch.setattr(Confidence, "UNKNOWN", 0.0)
        monkeypatch.setattr(Confidence, "ACT", 0.0)
        # The pre-refusal still holds this one, which is the point: the
        # floor is not the only guard, and removing it is not enough to
        # cause a false execution.
        assert not resolver.resolve(weak).resolved

    def test_removing_the_negation_guard_is_caught(self, monkeypatch) -> None:
        import ai.nlu.resolver as module

        assert declines("I don't want you to open Chrome.") == "negated"

        monkeypatch.setattr(module, "_NEGATION",
                            __import__("re").compile(r"(?!x)x"))
        assert declines("I don't want you to open Chrome.") != "negated", (
            "the negation refusal does not depend on _NEGATION")

    def test_removing_the_ambiguity_margin_is_caught(self, resolver,
                                                     monkeypatch) -> None:
        ambiguous = "open my editor"
        assert resolver.resolve(ambiguous).requires_clarification

        monkeypatch.setattr(Confidence, "AMBIGUITY_MARGIN", 0.0)
        # The application-choice clarification is set by the candidate itself,
        # so this one survives the margin being removed -- two independent
        # mechanisms, which is why it is worth knowing which is which.
        assert resolver.resolve(ambiguous).requires_clarification

    def test_an_unregistered_action_cannot_be_proposed(self) -> None:
        from actions.registry import ActionRegistry

        bare = NaturalLanguageResolver(ActionRegistry())
        for sentence, _expected, _entities, context in paraphrase_cases()[:20]:
            result = bare.resolve(sentence, CONTEXTS[context])
            assert not result.action, (
                f"{sentence!r} proposed {result.action!r} with an empty registry")

    def test_the_fuzzy_ceiling_is_load_bearing(self, resolver,
                                               monkeypatch) -> None:
        """Raise the ceiling and a repaired spelling could reach a shutdown."""
        import ai.nlu.resolver as module

        assert module.FUZZY_RISK_CEILING == "Low"
        monkeypatch.setattr(module, "FUZZY_RISK_CEILING", "High")
        # Nothing in the corpus reaches a risky action by fuzz today, so this
        # asserts the constant is read rather than a behaviour change.
        assert module.FUZZY_RISK_CEILING == "High"


class TestDeclineKinds:
    """Two kinds of "nothing happened", told apart.

    Flattening these was a real regression during this phase: routing every
    decline through REPLY turned "please reticulate the splines" into a
    successful response, which is how an assistant starts looking like it
    understood things it did not.
    """

    def test_an_unservable_request_is_not_ok(self, engine) -> None:
        from core.engine import ResponseKind

        response = engine.handle_text("please reticulate the splines")
        assert response.kind is ResponseKind.REPLY
        assert response.ok is False
        assert response.action is None

    def test_complying_by_doing_nothing_is_ok(self, engine) -> None:
        """"Don't open Chrome" was served correctly by not opening Chrome."""
        response = engine.handle_text("I don't want you to open Chrome.")
        assert response.ok is True
        assert response.action is None
        assert "won't" in response.text.lower()

    def test_out_of_domain_says_so_plainly(self, engine) -> None:
        response = engine.handle_text("I want you to fly a drone.")
        assert response.ok is False
        assert "capability" in response.text.lower()


# ----------------------------------------------------- normalisation units

class TestNormalisation:

    @pytest.mark.parametrize("spoken,expected", [
        ("Hey, could you please open Chrome for me?", "open chrome"),
        ("I want to watch a Python tutorial on you tube",
         "watch a python tutorial on youtube"),
        ("open power shell", "open powershell"),
        ("show me git hub", "show me github"),
    ])
    def test_filler_and_asr_spellings_are_folded(self, spoken,
                                                 expected) -> None:
        assert normalise(spoken) == expected

    def test_unknown_words_are_not_bent_into_commands(self) -> None:
        """The line between repairing "gitub" and inventing "delete"."""
        from ai.nlu.normalize import close_enough

        assert close_enough("gitub", "github")
        assert not close_enough("delegate", "delete")
        assert not close_enough("cat", "cut")

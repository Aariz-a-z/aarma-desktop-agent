"""Permissions, risk policy, the filesystem sandbox, confirmations and passphrases.

Nothing here touches a real user folder: the sandbox fixture is rooted in a
pytest temporary directory, and the destructive-path tests assert on *refusals*
rather than performing deletions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.settings import ConfigManager, SecuritySettings
from security.confirmation import ConfirmationBroker, ConfirmationOutcome, interpret_answer
from security.passphrase import (
    PassphraseError,
    PassphraseManager,
    hash_passphrase,
    verify_passphrase,
)
from security.permissions import (
    PERMISSION_CATALOG,
    Permission,
    PermissionManager,
)
from security.validator import (
    RiskLevel,
    RiskPolicy,
    SafePathResolver,
    UnsafeInputError,
    is_safe_url,
    normalize_url,
    sanitize_component,
)
from utils.logger import SecretRedactingFilter, redact


class TestPermissions:
    def test_observation_capabilities_are_denied_by_default(
        self, permissions: PermissionManager
    ) -> None:
        for permission in (Permission.MICROPHONE, Permission.CAMERA,
                           Permission.SCREENSHOT, Permission.KEYBOARD,
                           Permission.MOUSE, Permission.SYSTEM_CONTROL):
            assert not permissions.is_granted(permission)

    def test_everyday_capabilities_are_granted_by_default(
        self, permissions: PermissionManager
    ) -> None:
        for permission in (Permission.BROWSER, Permission.FILESYSTEM, Permission.INTERNET):
            assert permissions.is_granted(permission)

    def test_grant_and_revoke_round_trip(self, permissions: PermissionManager) -> None:
        permissions.grant(Permission.MICROPHONE)
        assert permissions.is_granted(Permission.MICROPHONE)
        permissions.revoke(Permission.MICROPHONE)
        assert not permissions.is_granted(Permission.MICROPHONE)

    def test_check_reports_every_missing_permission(
        self, permissions: PermissionManager
    ) -> None:
        decision = permissions.check((Permission.CAMERA, Permission.MICROPHONE,
                                      Permission.BROWSER))
        assert not decision.allowed
        assert set(decision.missing) == {Permission.CAMERA, Permission.MICROPHONE}

    def test_check_passes_when_everything_is_granted(
        self, permissions: PermissionManager
    ) -> None:
        assert permissions.check((Permission.BROWSER,)).allowed
        assert permissions.check(()).allowed

    def test_permissions_persist_through_the_config(self, config: ConfigManager) -> None:
        PermissionManager(config).grant(Permission.MICROPHONE)
        assert config.config.permissions.granted["microphone"] is True

    def test_every_permission_has_catalogue_metadata(self) -> None:
        for permission in Permission:
            assert permission in PERMISSION_CATALOG
            assert PERMISSION_CATALOG[permission].description


class TestRiskPolicy:
    def test_risk_levels_are_ordered(self) -> None:
        assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH

    def test_high_risk_always_requires_confirmation(self) -> None:
        policy = RiskPolicy.from_settings(SecuritySettings())
        assert policy.requires_confirmation(RiskLevel.HIGH)
        assert not policy.requires_confirmation(RiskLevel.MEDIUM)
        assert not policy.requires_confirmation(RiskLevel.LOW)

    def test_confirmation_floor_cannot_be_switched_off_by_settings(self) -> None:
        relaxed = SecuritySettings(enabled=False, confirm_high_risk_actions=False)
        assert RiskPolicy.from_settings(relaxed).requires_confirmation(RiskLevel.HIGH)

    def test_passphrase_gate_is_opt_in(self) -> None:
        off = RiskPolicy.from_settings(SecuritySettings())
        assert not off.requires_passphrase(RiskLevel.HIGH)

        on = RiskPolicy.from_settings(SecuritySettings(
            enabled=True, passphrase_hash="x", passphrase_salt="y",
            require_passphrase_for_high_risk=True,
        ))
        assert on.requires_passphrase(RiskLevel.HIGH)
        assert not on.requires_passphrase(RiskLevel.MEDIUM)


class TestPathSandbox:
    def test_resolves_a_child_inside_the_sandbox(self, resolver: SafePathResolver,
                                                 sandbox: Path) -> None:
        assert resolver.resolve_child(None, "New Folder").parent == sandbox

    @pytest.mark.parametrize("hostile", [
        "../escape", "..\\escape", "sub/dir", "sub\\dir", "..", ".",
    ])
    def test_traversal_in_a_name_is_refused(self, resolver: SafePathResolver,
                                            hostile: str) -> None:
        with pytest.raises(UnsafeInputError):
            resolver.resolve_child(None, hostile)

    @pytest.mark.parametrize("hostile", [
        "C:/Windows/System32", "C:\\Windows", "/etc", "..", "../..",
    ])
    def test_absolute_paths_outside_the_sandbox_are_refused(
        self, resolver: SafePathResolver, hostile: str
    ) -> None:
        with pytest.raises(UnsafeInputError):
            resolver.resolve_base(hostile)

    def test_existing_path_outside_the_sandbox_is_refused(
        self, resolver: SafePathResolver, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(UnsafeInputError):
            resolver.resolve_existing(str(outside))
        assert outside.exists()  # Nothing was touched.

    def test_existing_path_inside_the_sandbox_resolves(
        self, resolver: SafePathResolver, sandbox: Path
    ) -> None:
        assert resolver.resolve_existing("notes.txt") == sandbox / "notes.txt"

    def test_missing_path_is_reported_not_created(self, resolver: SafePathResolver) -> None:
        with pytest.raises(UnsafeInputError):
            resolver.resolve_existing("no-such-file.txt")

    @pytest.mark.parametrize("reserved", ["CON", "nul", "COM1", "LPT9.txt"])
    def test_windows_reserved_device_names_are_refused(self, reserved: str) -> None:
        with pytest.raises(UnsafeInputError):
            sanitize_component(reserved)

    @pytest.mark.parametrize("bad", ['a<b', 'a>b', 'a:b', 'a"b', "a|b", "a?b", "a*b"])
    def test_illegal_filename_characters_are_refused(self, bad: str) -> None:
        with pytest.raises(UnsafeInputError):
            sanitize_component(bad)

    def test_ordinary_names_are_accepted(self) -> None:
        assert sanitize_component("  AI Project  ") == "AI Project"


class TestUrlSafety:
    @pytest.mark.parametrize("url", [
        "https://example.com", "http://example.com/path?q=1",
    ])
    def test_web_urls_are_safe(self, url: str) -> None:
        assert is_safe_url(url)

    @pytest.mark.parametrize("url", [
        "file:///C:/Windows/System32/cmd.exe",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "ms-settings:",
        "vbscript:msgbox",
        "not a url at all",
    ])
    def test_non_web_schemes_are_unsafe(self, url: str) -> None:
        assert not is_safe_url(url)

    def test_bare_hostname_becomes_https(self) -> None:
        assert normalize_url("youtube.com") == "https://youtube.com"

    @pytest.mark.parametrize("value", ["", "chrome", "file:///etc/passwd", "notepad"])
    def test_non_addresses_are_refused(self, value: str) -> None:
        with pytest.raises(UnsafeInputError):
            normalize_url(value)


class TestConfirmations:
    def test_approval_returns_the_original_request(self) -> None:
        broker = ConfirmationBroker()
        request = broker.create("delete_item", "Delete a file", {"path": "notes.txt"})
        outcome, resolved = broker.resolve(request.token, True)
        assert outcome is ConfirmationOutcome.APPROVED
        assert resolved is not None and resolved.parameters == {"path": "notes.txt"}

    def test_declining_reports_declined(self) -> None:
        broker = ConfirmationBroker()
        request = broker.create("delete_item", "Delete a file")
        outcome, _ = broker.resolve(request.token, False)
        assert outcome is ConfirmationOutcome.DECLINED

    def test_a_token_can_only_be_used_once(self) -> None:
        broker = ConfirmationBroker()
        request = broker.create("delete_item", "Delete a file")
        broker.resolve(request.token, True)
        outcome, _ = broker.resolve(request.token, True)
        assert outcome is ConfirmationOutcome.UNKNOWN

    def test_unknown_token_is_refused(self) -> None:
        outcome, _ = ConfirmationBroker().resolve("not-a-real-token", True)
        assert outcome is ConfirmationOutcome.UNKNOWN

    def test_expired_requests_cannot_be_approved(self) -> None:
        broker = ConfirmationBroker(ttl_seconds=-1)
        request = broker.create("delete_item", "Delete a file")
        outcome, _ = broker.resolve(request.token, True)
        assert outcome is ConfirmationOutcome.EXPIRED

    def test_tokens_are_unguessable(self) -> None:
        broker = ConfirmationBroker()
        tokens = {broker.create("delete_item", "x").token for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(token) >= 16 for token in tokens)

    @pytest.mark.parametrize(("text", "expected"), [
        ("yes", True), ("Y", True), ("go ahead", True), ("confirm", True),
        ("no", False), ("cancel", False), ("never mind", False),
        ("maybe", None), ("delete everything", None), ("", None),
    ])
    def test_answer_interpretation(self, text: str, expected) -> None:
        assert interpret_answer(text) is expected


class TestPassphrase:
    def test_hash_is_salted_and_verifies(self) -> None:
        record = hash_passphrase("correct horse battery")
        assert verify_passphrase("correct horse battery", record)
        assert not verify_passphrase("wrong passphrase", record)

    def test_same_passphrase_hashes_differently_each_time(self) -> None:
        first = hash_passphrase("correct horse battery")
        second = hash_passphrase("correct horse battery")
        assert first.digest_hex != second.digest_hex
        assert first.salt_hex != second.salt_hex

    def test_short_passphrases_are_refused(self) -> None:
        with pytest.raises(PassphraseError):
            hash_passphrase("short")

    def test_plaintext_is_never_written_to_the_config(self, config: ConfigManager) -> None:
        secret = "correct horse battery"
        PassphraseManager(config).enable(secret)
        stored = config.path.read_text(encoding="utf-8")
        assert secret not in stored
        assert config.config.security.passphrase_hash

    def test_lock_and_unlock_cycle(self, config: ConfigManager) -> None:
        manager = PassphraseManager(config)
        manager.enable("correct horse battery")
        manager.lock()
        assert not manager.is_unlocked
        assert not manager.unlock("wrong passphrase")
        assert not manager.is_unlocked
        assert manager.unlock("correct horse battery")
        assert manager.is_unlocked

    def test_changing_requires_the_current_passphrase(self, config: ConfigManager) -> None:
        manager = PassphraseManager(config)
        manager.enable("correct horse battery")
        with pytest.raises(PassphraseError):
            manager.change("guessing", "a brand new phrase")
        manager.change("correct horse battery", "a brand new phrase")
        assert manager.verify("a brand new phrase")

    def test_disabling_requires_the_current_passphrase(self, config: ConfigManager) -> None:
        manager = PassphraseManager(config)
        manager.enable("correct horse battery")
        with pytest.raises(PassphraseError):
            manager.disable("guessing")
        manager.disable("correct horse battery")
        assert not manager.is_enabled

    def test_security_starts_locked_after_a_restart(self, config: ConfigManager) -> None:
        PassphraseManager(config).enable("correct horse battery")
        restarted = PassphraseManager(config)
        assert restarted.is_enabled
        assert not restarted.is_unlocked


class TestLogRedaction:
    @pytest.mark.parametrize("key", ["password", "passphrase", "api_key", "token", "secret"])
    def test_sensitive_mapping_keys_are_redacted(self, key: str) -> None:
        assert redact({key: "hunter2"})[key] == "***REDACTED***"

    def test_nested_structures_are_redacted(self) -> None:
        payload = {"outer": {"api_key": "sk-12345"}, "items": [{"token": "abc"}]}
        cleaned = redact(payload)
        assert cleaned["outer"]["api_key"] == "***REDACTED***"
        assert cleaned["items"][0]["token"] == "***REDACTED***"

    def test_inline_secrets_in_free_text_are_redacted(self) -> None:
        assert "hunter2" not in redact("logging in with password=hunter2 now")

    def test_ordinary_values_survive(self) -> None:
        assert redact({"action": "get_time", "count": 3}) == {"action": "get_time", "count": 3}

    def test_filter_scrubs_a_log_record(self) -> None:
        record = logging.LogRecord("aarma.test", logging.INFO, __file__, 1,
                                   "unlocking with passphrase=hunter2", None, None)
        record.context = {"api_key": "sk-secret"}  # type: ignore[attr-defined]
        SecretRedactingFilter().filter(record)
        assert "hunter2" not in record.msg
        assert record.context["api_key"] == "***REDACTED***"  # type: ignore[attr-defined]

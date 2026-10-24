"""End-to-end tests of the pipeline, including safe action execution.

These run against a real engine whose filesystem sandbox is a pytest temporary
directory and whose network mode is forced offline, so no test can reach a real
document, launch a real application or open a real browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from actions.registry import ActionRequest
from core.engine import AarmaEngine, ResponseKind
from security.permissions import Permission


class TestConversation:
    def test_greeting_uses_the_current_assistant_name(self, engine: AarmaEngine) -> None:
        engine.config.update_assistant(name="Nova")
        assert "Nova" in engine.handle_text("hello").text

    def test_time_and_date_run_without_any_permission(self, engine: AarmaEngine) -> None:
        for utterance in ("what time is it", "what's the date"):
            response = engine.handle_text(utterance)
            assert response.kind is ResponseKind.RESULT
            assert response.ok

    def test_unknown_input_is_answered_not_guessed(self, engine: AarmaEngine) -> None:
        response = engine.handle_text("please reticulate the splines")
        assert response.kind is ResponseKind.REPLY
        assert not response.ok
        assert response.action is None

    def test_exit_signals_shutdown(self, engine: AarmaEngine) -> None:
        assert engine.handle_text("goodbye").should_exit

    def test_conversation_is_recorded(self, engine: AarmaEngine) -> None:
        engine.handle_text("what time is it")
        roles = [row["role"] for row in engine.database.recent_messages()]
        assert roles[-2:] == ["user", "assistant"]

    def test_handle_text_never_raises(self, engine: AarmaEngine) -> None:
        for hostile in ("", "   ", "\x00", "*" * 3000, "'; DROP TABLE conversation; --"):
            assert engine.handle_text(hostile) is not None


class TestPermissionEnforcement:
    def test_denied_permission_blocks_the_action(self, engine: AarmaEngine) -> None:
        engine.permissions.revoke(Permission.FILESYSTEM)
        response = engine.handle_text("list files in Documents")
        assert not response.ok
        assert "permission" in response.text.lower()

    def test_granting_the_permission_unblocks_it(self, engine: AarmaEngine,
                                                 sandbox: Path) -> None:
        engine.permissions.grant(Permission.FILESYSTEM)
        response = engine.execute(ActionRequest(action="list_files", parameters={}))
        assert response.ok
        assert "notes.txt" in response.text

    def test_denial_is_written_to_the_audit_trail(self, engine: AarmaEngine) -> None:
        engine.permissions.revoke(Permission.FILESYSTEM)
        engine.execute(ActionRequest(action="list_files", parameters={}))
        rows = engine.database.recent_permission_decisions()
        assert any(row["permission"] == "filesystem" and row["decision"] == "denied"
                   for row in rows)


class TestOfflineHandling:
    def test_online_actions_are_refused_while_offline(self, engine: AarmaEngine) -> None:
        response = engine.execute(
            ActionRequest(action="search_web", parameters={"query": "nvidia"})
        )
        assert not response.ok
        assert "offline" in response.text.lower()

    def test_local_actions_still_work_offline(self, engine: AarmaEngine) -> None:
        assert engine.execute(ActionRequest(action="get_time")).ok


class TestUnregisteredActions:
    @pytest.mark.parametrize("name", ["run_shell_command", "delete_everything", "sudo"])
    def test_unregistered_names_are_refused(self, engine: AarmaEngine, name: str) -> None:
        response = engine.execute(ActionRequest(action=name))
        assert not response.ok
        assert response.kind is ResponseKind.ERROR

    def test_undeclared_parameters_are_refused(self, engine: AarmaEngine) -> None:
        response = engine.execute(
            ActionRequest(action="get_time", parameters={"shell": "cmd.exe"})
        )
        assert not response.ok


class TestSafeActionExecution:
    def test_create_folder_inside_the_sandbox(self, engine: AarmaEngine,
                                              sandbox: Path) -> None:
        response = engine.execute(
            ActionRequest(action="create_folder", parameters={"name": "AI Project"})
        )
        assert response.ok
        assert (sandbox / "AI Project").is_dir()

    def test_create_folder_refuses_traversal(self, engine: AarmaEngine,
                                             sandbox: Path) -> None:
        response = engine.execute(
            ActionRequest(action="create_folder", parameters={"name": "../escaped"})
        )
        assert not response.ok
        assert not (sandbox.parent / "escaped").exists()

    def test_creating_an_existing_folder_is_reported_not_overwritten(
        self, engine: AarmaEngine, sandbox: Path
    ) -> None:
        (sandbox / "Reports" / "keep.txt").write_text("keep me", encoding="utf-8")
        response = engine.execute(
            ActionRequest(action="create_folder", parameters={"name": "Reports"})
        )
        assert not response.ok
        assert "already exists" in response.text.lower()
        assert (sandbox / "Reports" / "keep.txt").read_text(encoding="utf-8") == "keep me"

    def test_rename_inside_the_sandbox(self, engine: AarmaEngine, sandbox: Path) -> None:
        response = engine.execute(ActionRequest(
            action="rename_item", parameters={"path": "notes.txt", "new_name": "renamed.txt"}
        ))
        assert response.ok
        assert (sandbox / "renamed.txt").exists()
        assert not (sandbox / "notes.txt").exists()

    def test_open_file_refuses_executables(self, engine: AarmaEngine,
                                           sandbox: Path) -> None:
        (sandbox / "installer.exe").write_bytes(b"MZ")
        response = engine.execute(
            ActionRequest(action="open_file", parameters={"name": "installer.exe"})
        )
        assert not response.ok
        assert response.data.get("path") is None

    def test_open_application_refuses_anything_off_the_catalogue(
        self, engine: AarmaEngine
    ) -> None:
        engine.permissions.grant(Permission.APPLICATIONS)
        response = engine.execute(
            ActionRequest(action="open_application",
                          parameters={"name": "definitely not installed"})
        )
        assert not response.ok
        assert "approved" in response.text.lower()

    def test_open_application_refuses_a_shell(self, engine: AarmaEngine) -> None:
        """A shell is on the catalogue, but not through this action.

        ``powershell`` used to be off the catalogue entirely, which is what this
        test originally asserted.  Phase 3 added the terminals behind their own
        action and their own permission, so the refusal here is now about
        *routing* -- and it still has to be a refusal, because reaching a shell
        through the permission set meant for Notepad is the thing to prevent.
        """
        engine.permissions.grant(Permission.APPLICATIONS)
        response = engine.execute(
            ActionRequest(action="open_application", parameters={"name": "powershell"})
        )
        assert not response.ok
        assert response.data.get("error_code") == "use_open_terminal" or \
            "terminal" in response.text.lower()

    def test_open_application_launches_only_a_resolved_path(
        self, engine: AarmaEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launched: list = []
        # Two arguments since Phase 3: the target and the catalogue-supplied
        # argv. Nothing user-spoken reaches either, which is what the second
        # assertion below pins down.
        monkeypatch.setattr("actions.applications._launch",
                            lambda target, arguments=(): launched.append(
                                (target, tuple(arguments))))
        monkeypatch.setattr("actions.applications.resolve_target",
                            lambda entry: r"C:\fake\notepad.exe")
        engine.permissions.grant(Permission.APPLICATIONS)

        response = engine.handle_text("open notepad")
        assert response.ok
        assert launched == [(r"C:\fake\notepad.exe", ())]

    def test_web_search_builds_an_encoded_url(self, engine: AarmaEngine,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list[str] = []
        monkeypatch.setattr("actions.browser._open_in_browser",
                            lambda url: opened.append(url) or True)
        engine.config.update_runtime(network_mode="online")
        engine.connectivity.invalidate()

        response = engine.handle_text("search for python & rust")
        assert response.ok
        assert opened and "python+%26+rust" in opened[0]


class TestConfirmationFlow:
    def _park_a_deletion(self, engine: AarmaEngine, sandbox: Path):
        (sandbox / "disposable.txt").write_text("bye", encoding="utf-8")
        return engine.execute(
            ActionRequest(action="delete_item", parameters={"path": "disposable.txt"})
        )

    def test_high_risk_action_asks_first(self, engine: AarmaEngine, sandbox: Path) -> None:
        response = self._park_a_deletion(engine, sandbox)
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert response.confirmation_token
        assert (sandbox / "disposable.txt").exists()  # Nothing happened yet.

    def test_declining_leaves_the_file_alone(self, engine: AarmaEngine,
                                             sandbox: Path) -> None:
        parked = self._park_a_deletion(engine, sandbox)
        result = engine.resolve_confirmation(parked.confirmation_token, False)
        assert result.ok
        assert (sandbox / "disposable.txt").exists()

    def test_approving_runs_the_action(self, engine: AarmaEngine, sandbox: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        removed: list[str] = []
        monkeypatch.setattr("send2trash.send2trash", removed.append)
        parked = self._park_a_deletion(engine, sandbox)
        result = engine.resolve_confirmation(parked.confirmation_token, True)
        assert result.ok
        assert removed and removed[0].endswith("disposable.txt")

    def test_a_bare_yes_answers_the_outstanding_confirmation(
        self, engine: AarmaEngine, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        removed: list[str] = []
        monkeypatch.setattr("send2trash.send2trash", removed.append)
        (sandbox / "disposable.txt").write_text("bye", encoding="utf-8")

        parked = engine.handle_text("delete disposable.txt")
        assert parked.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert engine.handle_text("yes").ok
        assert removed

    def test_a_bare_no_cancels(self, engine: AarmaEngine, sandbox: Path) -> None:
        (sandbox / "disposable.txt").write_text("bye", encoding="utf-8")
        engine.handle_text("delete disposable.txt")
        engine.handle_text("no")
        assert (sandbox / "disposable.txt").exists()

    def test_out_of_sandbox_target_is_refused_before_asking(
        self, engine: AarmaEngine
    ) -> None:
        """Never ask someone to approve something that will be refused anyway."""
        response = engine.execute(
            ActionRequest(action="delete_item",
                          parameters={"path": r"C:\Windows\System32\drivers\etc\hosts"})
        )
        assert response.kind is ResponseKind.ERROR
        assert response.confirmation_token is None


class TestSecurityLock:
    def test_a_locked_engine_answers_nothing_else(self, engine: AarmaEngine) -> None:
        engine.passphrase.enable("correct horse battery")
        engine.passphrase.lock()
        response = engine.handle_text("what time is it")
        assert not response.ok
        assert "locked" in response.text.lower()

    def test_the_passphrase_unlocks_it(self, engine: AarmaEngine) -> None:
        engine.passphrase.enable("correct horse battery")
        engine.passphrase.lock()
        assert engine.handle_text("correct horse battery").ok
        assert engine.handle_text("what time is it").ok


class TestAuditTrail:
    def test_successful_actions_are_recorded_with_their_risk(
        self, engine: AarmaEngine
    ) -> None:
        engine.handle_text("what time is it")
        entry = engine.database.recent_actions(limit=1)[0]
        assert entry.action == "get_time"
        assert entry.risk == "Low"
        assert entry.allowed and entry.succeeded

    def test_handler_exceptions_are_caught_and_recorded(
        self, engine: AarmaEngine
    ) -> None:
        """A crashing handler must produce a sentence, not a stack trace."""
        from actions.registry import ActionSpec
        from security.validator import RiskLevel

        def _boom(ctx, params):
            raise RuntimeError("database on fire")

        engine.registry.register(ActionSpec(
            name="faulty_action", description="Always raises.", category="Test",
            risk=RiskLevel.LOW, handler=_boom,
        ))

        response = engine.execute(ActionRequest(action="faulty_action"))
        assert not response.ok
        assert "Traceback" not in response.text
        entry = engine.database.recent_actions(limit=1)[0]
        assert entry.action == "faulty_action" and not entry.succeeded

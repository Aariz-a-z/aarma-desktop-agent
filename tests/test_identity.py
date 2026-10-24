"""Assistant identity: defaults, changes and persistence (specification section 4)."""

from __future__ import annotations

from pathlib import Path

from actions.assistant import get_actions
from config.settings import (
    ASSISTANT_PROFILES,
    AssistantIdentity,
    ConfigManager,
    load_config,
)


def _action(name: str):
    return next(spec for spec in get_actions() if spec.name == name)


class TestDefaults:
    def test_default_identity_is_asma(self, config: ConfigManager) -> None:
        assistant = config.config.assistant
        assert assistant.name == "Asma"
        assert assistant.gender == "female"
        assert assistant.profile_key == "asma"

    def test_both_shipped_profiles_exist(self) -> None:
        assert set(ASSISTANT_PROFILES) == {"asma", "aariz"}
        assert ASSISTANT_PROFILES["aariz"].gender == "male"

    def test_default_wake_word_matches_default_name(self, config: ConfigManager) -> None:
        from config.settings import wake_phrase_for_name

        assistant = config.config.assistant
        assert assistant.wake_word == wake_phrase_for_name(assistant.name)
        # The phrase is the name itself: people say "Asma", not "hey Asma".
        assert assistant.wake_word == assistant.name.lower()


class TestChangingIdentity:
    def test_rename_updates_the_stored_name(self, config: ConfigManager) -> None:
        config.update_assistant(name="Nova")
        assert config.config.assistant.name == "Nova"

    def test_switching_profile_replaces_voice_and_wake_word(self, config: ConfigManager) -> None:
        config.update(assistant=AssistantIdentity.from_profile(ASSISTANT_PROFILES["aariz"]))
        assistant = config.config.assistant
        assert assistant.name == "Aariz"
        assert assistant.voice_id == "male_calm"
        assert assistant.wake_word == "hey aariz"

    def test_rename_action_moves_the_wake_word_with_the_name(self, action_context) -> None:
        result = _action("set_assistant_name").handler(action_context, {"name": "Nova"})
        assert result.ok
        assistant = action_context.config.config.assistant
        assert assistant.name == "Nova"
        assert assistant.wake_word == "nova"
        assert assistant.profile_key == "custom"

    def test_rename_keeps_a_custom_wake_word(self, action_context) -> None:
        action_context.config.update_assistant(wake_word="computer")
        _action("set_assistant_name").handler(action_context, {"name": "Nova"})
        assert action_context.config.config.assistant.wake_word == "computer"

    def test_rename_rejects_a_name_with_no_letters(self, action_context) -> None:
        result = _action("set_assistant_name").handler(action_context, {"name": "1234"})
        assert not result.ok
        assert action_context.config.config.assistant.name == "Asma"

    def test_rename_rejects_an_empty_name(self, action_context) -> None:
        result = _action("set_assistant_name").handler(action_context, {"name": "   "})
        assert not result.ok

    def test_unknown_profile_is_refused(self, action_context) -> None:
        result = _action("set_assistant_profile").handler(action_context, {"profile": "zephyr"})
        assert not result.ok
        assert result.error_code == "unknown_profile"


class TestPersistence:
    def test_name_survives_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        first = ConfigManager(path)
        first.load()
        first.update_assistant(name="Nova", wake_word="hey nova")

        reloaded = load_config(path)
        assert reloaded.config.assistant.name == "Nova"
        assert reloaded.config.assistant.wake_word == "hey nova"

    def test_corrupt_config_falls_back_to_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{ this is not json", encoding="utf-8")
        manager = load_config(path)
        assert manager.config.assistant.name == "Asma"

    def test_unknown_keys_are_ignored_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(
            '{"assistant": {"name": "Nova", "invented_field": 42}}', encoding="utf-8"
        )
        manager = load_config(path)
        assert manager.config.assistant.name == "Nova"

    def test_listeners_are_notified_on_change(self, config: ConfigManager) -> None:
        seen: list[str] = []
        config.add_listener(lambda snapshot: seen.append(snapshot.assistant.name))
        config.update_assistant(name="Iris")
        assert seen == ["Iris"]

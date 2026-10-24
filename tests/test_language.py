"""Language configuration: defaults, resolution, changes and persistence.

Specification section 6 requires interface, input and output languages to be
independent, and requires the choice to persist.  It also requires honesty
about which languages are actually translated, which is what the
:class:`~language.manager.LanguageStatus` assertions here pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import ConfigManager, load_config
from language.manager import (
    LANGUAGE_CATALOG,
    LanguageManager,
    LanguageStatus,
    resolve_language_code,
)


class TestCatalogue:
    def test_english_is_the_only_translated_language_in_this_build(self) -> None:
        translated = [info.code for info in LANGUAGE_CATALOG.values()
                      if info.status is LanguageStatus.TRANSLATED]
        assert translated == ["en"]

    def test_phase_two_indian_languages_are_declared(self) -> None:
        for code in ("hi", "kn", "te", "ta", "ml"):
            info = LANGUAGE_CATALOG[code]
            assert info.status is LanguageStatus.DECLARED
            assert info.phase == 2

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Hindi", "hi"), ("hindi", "hi"), ("hi", "hi"), ("hi-IN", "hi"),
            ("Kannada", "kn"), ("kannad", "kn"),
            ("English", "en"), ("en_GB", "en"),
            ("हिन्दी", "hi"),
        ],
    )
    def test_resolves_names_codes_and_aliases(self, value: str, expected: str) -> None:
        assert resolve_language_code(value) == expected

    @pytest.mark.parametrize("value", ["", "klingon", "xx", "not a language"])
    def test_unknown_languages_resolve_to_none(self, value: str) -> None:
        assert resolve_language_code(value) is None


class TestDefaults:
    def test_default_language_is_english_everywhere(self, config: ConfigManager) -> None:
        settings = config.config.language
        assert (settings.interface, settings.speech_input, settings.speech_output) == \
               ("en", "en", "en")

    def test_auto_detect_is_off_by_default(self, config: ConfigManager) -> None:
        assert config.config.language.auto_detect is False


class TestChangingLanguage:
    def test_change_updates_all_three_settings(self, language: LanguageManager) -> None:
        info = language.set_language("Hindi")
        assert info is not None and info.code == "hi"
        settings = language.config.config.language
        assert (settings.interface, settings.speech_input, settings.speech_output) == \
               ("hi", "hi", "hi")

    def test_scoped_change_touches_only_that_setting(self, language: LanguageManager) -> None:
        language.set_language("Tamil", scope="input")
        settings = language.config.config.language
        assert settings.speech_input == "ta"
        assert settings.interface == "en"
        assert settings.speech_output == "en"

    def test_unknown_language_is_refused_and_changes_nothing(
        self, language: LanguageManager
    ) -> None:
        assert language.set_language("Klingon") is None
        assert language.config.config.language.interface == "en"


class TestMessages:
    def test_renders_a_message_with_parameters(self, language: LanguageManager) -> None:
        assert language.t("response.name_changed", assistant="Nova") == \
               "Done. You can call me Nova now."

    def test_declared_language_falls_back_to_english_text(
        self, language: LanguageManager
    ) -> None:
        language.set_language("Hindi")
        rendered = language.t("response.goodbye")
        assert rendered == "Goodbye."  # Honest fallback, not a fabricated translation.

    def test_missing_key_returns_the_key_rather_than_raising(
        self, language: LanguageManager
    ) -> None:
        assert language.t("response.does_not_exist") == "response.does_not_exist"

    def test_missing_format_argument_does_not_raise(self, language: LanguageManager) -> None:
        assert "{assistant}" in language.t("response.greeting")


class TestPersistence:
    def test_language_survives_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        manager = ConfigManager(path)
        manager.load()
        LanguageManager(manager).set_language("Kannada")

        reloaded = load_config(path)
        assert reloaded.config.language.interface == "kn"
        assert LanguageManager(reloaded).display_name() == "Kannada"

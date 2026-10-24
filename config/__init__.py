"""Centralised configuration for AARMA.

Every user-adjustable value lives in :mod:`config.settings`.  Modules must read
their settings from the shared :class:`~config.settings.ConfigManager` rather
than hardcoding names, languages, paths or feature flags (specification
section 27).
"""

from config.settings import (
    AppConfig,
    AssistantIdentity,
    ASSISTANT_PROFILES,
    AssistantProfile,
    ConfigManager,
    LanguageSettings,
    PermissionSettings,
    RuntimeSettings,
    SecuritySettings,
    VoiceSettings,
)

__all__ = [
    "VoiceSettings",
    "AppConfig",
    "AssistantIdentity",
    "AssistantProfile",
    "ASSISTANT_PROFILES",
    "ConfigManager",
    "LanguageSettings",
    "PermissionSettings",
    "RuntimeSettings",
    "SecuritySettings",
]

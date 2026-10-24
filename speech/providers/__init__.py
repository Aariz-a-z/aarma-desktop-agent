"""Concrete text-to-speech backends.

Each provider implements :class:`~speech.speaker.SpeechSpeaker` and knows
nothing about the rest of AARMA. Two ship in Phase 2A:

``sapi``
    Offline. Windows' built-in voices, including the OneCore set that the
    classic interface hides. Always available on Windows; **standard** quality.

``edge``
    Online. Microsoft's neural voices, covering every language in AARMA's
    catalogue. **Neural** quality; needs a connection.

:func:`~speech.engine.SpeechEngine` picks between them per request and falls
back explicitly rather than silently.
"""

from speech.providers.edge import EdgeNeuralSpeaker
from speech.providers.sapi import SapiSpeaker

__all__ = ["EdgeNeuralSpeaker", "SapiSpeaker"]

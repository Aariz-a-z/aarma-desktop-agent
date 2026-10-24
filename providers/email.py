"""Email, with the line between *drafted* and *sent* drawn in the type system.

"Email sent." is the single most dangerous sentence an assistant can get wrong.
The user stops composing, closes the laptop, and finds out three days later
that nobody received anything.  So this module makes the distinction structural
rather than a matter of careful wording:

* :class:`LocalDraftEmailProvider` can write a draft into AARMA's own database.
  It has **no transport**, so :meth:`send` is not "unimplemented" -- it is a
  refusal with a reason, and no code path through it can report a send.
* A real account is represented by a different provider entirely.  Until one is
  connected, :func:`configured_provider` returns the local drafter, and every
  message it produces names AARMA's own drafts folder out loud.

Credentials are deliberately outside this file's scope.  AARMA does not collect
a password, does not read one from a config file, and does not touch the
browser's cookie store.  When a real provider is added it will authenticate
through the operating system's credential store or an OAuth flow the user
completes themselves -- and until that exists, saying email works would be the
lie this whole design is built to prevent.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from providers import Availability, Provider, ProviderResult, UnavailableProvider
from utils.logger import get_logger
from utils.when import local_now

logger = get_logger("providers.email")

MAX_SUBJECT = 200
MAX_BODY = 20000
MAX_RECIPIENTS = 20

#: Deliberately permissive: this checks the address is *shaped* like an address
#: so a transcription slip ("call mom at gmail") is caught, not that it exists.
#: Only a real mail server can tell you that, and pretending otherwise would be
#: the same category of error as pretending to send.
_ADDRESS = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]{2,}$")


def split_recipients(raw: str) -> List[str]:
    """Split a recipient string on the separators people actually speak."""
    parts = re.split(r"[,;]|\s+and\s+", str(raw or ""))
    return [part.strip() for part in parts if part.strip()]


def invalid_recipients(addresses: Sequence[str]) -> List[str]:
    return [address for address in addresses if not _ADDRESS.match(address)]


@dataclass(frozen=True)
class EmailDraft:
    """A message that exists but has not gone anywhere."""

    id: str
    to: List[str]
    subject: str
    body: str
    created_at: str
    provider: str = "local-drafts"
    #: Set only when a provider confirmed a send.  Empty means not sent, and
    #: nothing in this module sets it without that confirmation.
    sent_at: str = ""

    @property
    def sent(self) -> bool:
        return bool(self.sent_at)

    def summary(self) -> str:
        recipients = ", ".join(self.to) if self.to else "no recipient yet"
        subject = self.subject or "(no subject)"
        return f"\"{subject}\" to {recipients}"

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "to": list(self.to), "subject": self.subject,
                "body": self.body, "created_at": self.created_at,
                "provider": self.provider, "sent_at": self.sent_at,
                "sent": self.sent}


class EmailProvider(Provider):
    """What any mail account must offer."""

    name = "email"
    description = "draft and send email"

    #: Whether this provider has a transport at all.  Read by the capability
    #: report, so "what can you do?" cannot advertise sending that is absent.
    can_send: bool = False

    def create_draft(self, *, to: Sequence[str], subject: str,
                     body: str) -> ProviderResult:
        raise NotImplementedError

    def list_drafts(self, *, limit: int = 20) -> List[EmailDraft]:
        raise NotImplementedError

    def get_draft(self, draft_id: str) -> Optional[EmailDraft]:
        raise NotImplementedError

    def discard_draft(self, draft_id: str) -> ProviderResult:
        raise NotImplementedError

    def send(self, draft_id: str) -> ProviderResult:
        raise NotImplementedError


class LocalDraftEmailProvider(EmailProvider):
    """Drafts in AARMA's own database.  No transport, and it says so."""

    name = "local-drafts"
    description = "compose and keep drafts inside AARMA"
    can_send = False

    def __init__(self, database: Any) -> None:
        self._database = database

    def availability(self) -> Availability:
        if self._database is None:
            return Availability.missing(
                self.name, "AARMA's draft store isn't open.")
        # Available for drafting, and the detail says exactly which half works.
        return Availability.ready(
            self.name, "drafting only — no mail account is connected, so I "
                       "can write a message but not send it")

    # -- drafting ---------------------------------------------------------

    @staticmethod
    def _row_to_draft(row: Any) -> EmailDraft:
        recipients = [part for part in str(row["recipients"] or "").split(",")
                      if part]
        return EmailDraft(
            id=row["id"], to=recipients, subject=row["subject"] or "",
            body=row["body"] or "", created_at=row["created_at"] or "",
            provider=row["provider"] or "local-drafts",
            sent_at=row["sent_at"] or "")

    def create_draft(self, *, to: Sequence[str], subject: str,
                     body: str) -> ProviderResult:
        if self._database is None:
            return ProviderResult.unavailable(self.availability())
        recipients = [address.strip() for address in to if str(address).strip()]
        if len(recipients) > MAX_RECIPIENTS:
            return ProviderResult.failure(
                f"That's {len(recipients)} recipients; I'll draft to at most "
                f"{MAX_RECIPIENTS} at once.", error_code="too_many_recipients")
        bad = invalid_recipients(recipients)
        if bad:
            return ProviderResult.failure(
                f"{', '.join(bad)} doesn't look like an email address.",
                error_code="invalid_recipient", invalid=bad)
        if len(subject) > MAX_SUBJECT:
            return ProviderResult.failure("That subject line is too long.",
                                          error_code="subject_too_long")
        if len(body) > MAX_BODY:
            return ProviderResult.failure("That message is too long to draft.",
                                          error_code="body_too_long")

        draft_id = uuid.uuid4().hex[:12]
        try:
            self._database.add_email_draft(
                draft_id=draft_id, recipients=",".join(recipients),
                subject=subject, body=body, provider=self.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Draft could not be stored",
                         extra={"context": {"error": str(exc)}})
            return ProviderResult.failure("I couldn't save that draft.",
                                          error_code="storage_failed")
        # Recipients and body are not logged: the address book and the message
        # are the user's correspondence, not diagnostics.
        logger.info("Email draft created", extra={"context": {"id": draft_id}})
        return ProviderResult.success("Draft saved.", draft_id=draft_id,
                                      to=recipients, subject=subject)

    def list_drafts(self, *, limit: int = 20) -> List[EmailDraft]:
        if self._database is None:
            return []
        try:
            rows = self._database.email_drafts(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.error("Drafts could not be read",
                         extra={"context": {"error": str(exc)}})
            return []
        return [self._row_to_draft(row) for row in rows]

    def get_draft(self, draft_id: str) -> Optional[EmailDraft]:
        if self._database is None:
            return None
        try:
            row = self._database.email_draft(draft_id)
        except Exception:  # noqa: BLE001
            return None
        return self._row_to_draft(row) if row is not None else None

    def discard_draft(self, draft_id: str) -> ProviderResult:
        if self._database is None:
            return ProviderResult.unavailable(self.availability())
        if self._database.discard_email_draft(draft_id):
            return ProviderResult.success("Draft discarded.", draft_id=draft_id)
        return ProviderResult.failure("I couldn't find that draft.",
                                      error_code="not_found")

    # -- sending ----------------------------------------------------------

    def send(self, draft_id: str) -> ProviderResult:
        """Always a refusal.  There is no transport to succeed through.

        Written as an explicit method rather than left to inherit
        ``NotImplementedError`` so that the failure is a sentence the user can
        act on, and so that no caller can mistake an exception for a transport
        error that might work on a retry.
        """
        draft = self.get_draft(draft_id)
        detail = f" The draft ({draft.summary()}) is saved." if draft else ""
        return ProviderResult.failure(
            "I can't send email — no mail account is connected to AARMA." + detail
            + " Connecting one is a setup step that hasn't been built yet.",
            error_code="no_transport", draft_id=draft_id, sent=False)


def smtp_email() -> Provider:
    """Placeholder.  No SMTP account is configured in this build."""
    return UnavailableProvider(
        "smtp",
        "No outgoing mail server is configured.",
        "I can keep the message as a draft in AARMA instead.")


def gmail_email() -> Provider:
    """Placeholder.  Gmail is not connected in this build."""
    return UnavailableProvider(
        "gmail",
        "Gmail isn't connected.",
        "I can keep the message as a draft in AARMA instead.")


def outlook_email() -> Provider:
    """Placeholder.  Outlook mail is not connected in this build."""
    return UnavailableProvider(
        "outlook-mail",
        "Outlook mail isn't connected.",
        "I can keep the message as a draft in AARMA instead.")


def configured_provider(database: Any) -> EmailProvider:
    """The mail provider in use.

    One place decides, so that connecting a real account later changes this
    function and nothing else.  Today it is always the local drafter.
    """
    return LocalDraftEmailProvider(database)


__all__ = ["EmailDraft", "EmailProvider", "LocalDraftEmailProvider",
           "MAX_BODY", "MAX_RECIPIENTS", "MAX_SUBJECT", "configured_provider",
           "gmail_email", "invalid_recipients", "outlook_email",
           "smtp_email", "split_recipients"]

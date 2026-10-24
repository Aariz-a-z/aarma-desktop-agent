"""Email actions: draft, list, discard -- and a send that cannot lie.

The interesting design here is ``send_email``, and it is worth being explicit
about because it looks like a mistake at first glance: the action is registered
even though no provider in this build can send anything.

That is deliberate.  Leaving it out would make "send this email" an *unknown
command*, which tells the user nothing about why.  Registered, it produces the
real reason -- no account is connected -- and it does so *after* passing the
same HIGH-risk confirmation gate as any real send would.  So the confirmation
path is exercised, and the day a provider is connected the only change is that
:meth:`~providers.email.EmailProvider.send` starts succeeding.

What it will never do is report a send it did not make.  ``send_email`` reports
success only when a provider returns ``ok``, and the local drafter's
:meth:`send` is a refusal by construction rather than an unimplemented method
that might one day return the wrong thing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from providers.email import (
    MAX_BODY,
    MAX_SUBJECT,
    EmailProvider,
    configured_provider,
    split_recipients,
)
from security.permissions import Permission
from security.validator import RiskLevel, UnsafeInputError, ensure_text
from utils.logger import get_logger

logger = get_logger("actions.email")

MAX_LISTED_DRAFTS = 10


def provider_for(ctx: ActionContext) -> EmailProvider:
    """The mail provider in use, overridable by a test."""
    injected = getattr(ctx, "email_provider", None)
    if injected is not None:
        return injected
    return configured_provider(ctx.database)


def _unavailable(provider: EmailProvider) -> ActionResult:
    availability = provider.availability()
    text = availability.detail or "Email isn't available."
    if availability.remedy:
        text = f"{text} {availability.remedy}"
    return ActionResult.failure(text, error_code="email_unavailable")


def _draft(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)

    recipients = split_recipients(str(params.get("to") or ""))
    if not recipients:
        return ActionResult.failure("Who should I send it to?",
                                    error_code="missing_recipient", needs="to")
    subject = str(params.get("subject") or "").strip()
    if not subject:
        return ActionResult.failure(
            f"What's the subject for the message to {', '.join(recipients)}?",
            error_code="missing_subject", needs="subject")
    body = str(params.get("body") or "").strip()
    if not body:
        return ActionResult.failure(f"What should the message say?",
                                    error_code="missing_body", needs="body")

    try:
        subject = ensure_text(subject, field="subject", max_length=MAX_SUBJECT)
        body = ensure_text(body, field="message", max_length=MAX_BODY)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="invalid_text")

    result = provider.create_draft(to=recipients, subject=subject, body=body)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "draft_failed",
                                    **result.data)
    # "Saved as a draft in AARMA" rather than "saved": the user needs to know
    # this did not land in the drafts folder of the mail app they actually use.
    where = ("AARMA's own drafts" if provider.name == "local-drafts"
             else provider.name)
    tail = ("" if provider.can_send else
            " I can't send it — no mail account is connected.")
    return ActionResult.success(
        f"Saved a draft to {', '.join(recipients)} in {where}: \"{subject}\"."
        + tail, **result.data)


def _list_drafts(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    drafts = provider.list_drafts(limit=MAX_LISTED_DRAFTS)
    if not drafts:
        return ActionResult.success("You have no email drafts.",
                                    drafts=[], count=0)
    listed = "; ".join(draft.summary() for draft in drafts)
    heading = ("You have one draft: " if len(drafts) == 1
               else f"You have {len(drafts)} drafts: ")
    return ActionResult.success(heading + listed + ".",
                                drafts=[d.as_dict() for d in drafts],
                                count=len(drafts))


def _discard_draft(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    wanted = str(params.get("which") or "").strip().lower()
    drafts = provider.list_drafts(limit=MAX_LISTED_DRAFTS)
    if not drafts:
        return ActionResult.failure("There are no drafts to discard.",
                                    error_code="nothing_to_discard")
    matches = [d for d in drafts
               if not wanted or wanted == d.id or wanted in d.subject.lower()
               or any(wanted in address.lower() for address in d.to)]
    if not matches:
        return ActionResult.failure(f"No draft matches \"{wanted}\".",
                                    error_code="no_match")
    if len(matches) > 1:
        listed = "; ".join(d.summary() for d in matches[:4])
        return ActionResult.failure(
            f"{len(matches)} drafts match that: {listed}. Which one?",
            error_code="ambiguous")
    result = provider.discard_draft(matches[0].id)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "discard_failed")
    return ActionResult.success(f"Discarded the draft {matches[0].summary()}.",
                                **result.data)


def _send(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Send a draft -- if, and only if, a provider confirms it."""
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)

    wanted = str(params.get("which") or "").strip().lower()
    drafts = provider.list_drafts(limit=MAX_LISTED_DRAFTS)
    if not drafts:
        return ActionResult.failure(
            "There's no draft to send. Ask me to draft one first.",
            error_code="nothing_to_send")
    matches = [d for d in drafts
               if not wanted or wanted == d.id or wanted in d.subject.lower()
               or any(wanted in address.lower() for address in d.to)]
    if not matches:
        return ActionResult.failure(f"No draft matches \"{wanted}\".",
                                    error_code="no_match")
    if len(matches) > 1:
        listed = "; ".join(d.summary() for d in matches[:4])
        return ActionResult.failure(
            f"{len(matches)} drafts match that: {listed}. Which one?",
            error_code="ambiguous")

    draft = matches[0]
    result = provider.send(draft.id)
    if not result.ok:
        # Not an exception and not a silent success: the reason is the answer.
        logger.info("Email send refused",
                    extra={"context": {"reason": result.error_code}})
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "send_failed",
                                    sent=False)
    logger.info("Email sent", extra={"context": {"provider": provider.name}})
    return ActionResult.success(
        f"Sent {draft.summary()} via {provider.name}.", sent=True,
        **result.data)


def _email_status(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Say plainly what the email integration can and cannot do."""
    provider = provider_for(ctx)
    availability = provider.availability()
    if not availability.available:
        return ActionResult.failure(
            availability.detail, error_code="email_unavailable")
    if provider.can_send:
        return ActionResult.success(
            f"Email is connected through {provider.name}: I can draft and send.",
            provider=provider.name, can_send=True)
    return ActionResult.success(
        "No mail account is connected. I can write and keep drafts inside "
        "AARMA, but I can't send anything.",
        provider=provider.name, can_send=False)


def get_actions() -> List[ActionSpec]:
    which = ParameterSpec(
        name="which", type=ParameterType.STRING,
        description="which draft — a subject, a recipient, or blank for the only one",
        required=False, default="", max_length=120)
    return [
        ActionSpec(
            name="draft_email",
            description="Write an email and keep it as a draft.",
            category="Email", risk=RiskLevel.MEDIUM, handler=_draft,
            permissions=(Permission.EMAIL,),
            parameters=(
                ParameterSpec(name="to", description="who it goes to",
                              required=False, default="", max_length=400),
                ParameterSpec(name="subject", description="the subject line",
                              required=False, default="", max_length=MAX_SUBJECT),
                ParameterSpec(name="body", type=ParameterType.TEXT,
                              description="what the message says",
                              required=False, default="", max_length=MAX_BODY),
            ),
            examples=("draft an email to sam@example.com",
                      "write an email to my tutor"),
        ),
        ActionSpec(
            name="list_drafts", description="Show the email drafts you have.",
            category="Email", risk=RiskLevel.LOW, handler=_list_drafts,
            permissions=(Permission.EMAIL,),
            examples=("show my drafts", "what email drafts do I have"),
        ),
        ActionSpec(
            name="discard_draft", description="Throw away an email draft.",
            category="Email", risk=RiskLevel.MEDIUM, handler=_discard_draft,
            permissions=(Permission.EMAIL,), parameters=(which,),
            examples=("discard that draft", "delete the draft about the report"),
        ),
        ActionSpec(
            name="send_email",
            description="Send an email draft.",
            category="Email",
            # HIGH, and it stays HIGH once a real provider is connected: a sent
            # message cannot be recalled, and the recipient list is exactly the
            # thing a speech recogniser is most likely to have got wrong.
            risk=RiskLevel.HIGH, handler=_send,
            permissions=(Permission.EMAIL, Permission.INTERNET),
            parameters=(which,),
            examples=("send this email", "send the draft to sam"),
        ),
        ActionSpec(
            name="email_status",
            description="Say whether email is connected.",
            category="Email", risk=RiskLevel.LOW, handler=_email_status,
            examples=("is email set up", "can you send email"),
        ),
    ]

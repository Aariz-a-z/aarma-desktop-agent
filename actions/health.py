"""Actions that report what the machine is doing.

These are read-only and graded LOW: looking at a CPU meter changes nothing.
Killing a process is not here -- that is destructive and lives with the other
confirmed actions.

The one rule worth restating: a reading that could not be taken is reported as
unknown. :class:`~providers.system.SystemHealth` carries ``None`` for anything
unmeasured, and the phrasing below simply omits it rather than guessing.
"""

from __future__ import annotations

from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from providers.system import SYSTEM
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.health")


def _provider(ctx: ActionContext):
    """The system provider, overridable by a test through the context."""
    return getattr(ctx, "system_provider", None) or SYSTEM


def _unavailable(provider) -> ActionResult:
    availability = provider.availability()
    text = availability.detail or "I can't read this machine's vitals."
    if availability.remedy:
        text = f"{text} {availability.remedy}"
    return ActionResult.failure(text, error_code="system_unavailable")


def _system_health(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    return ActionResult.success(health.summary(), **health.as_dict())


def _cpu_usage(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    if health.cpu_percent is None:
        return ActionResult.failure("I couldn't read the CPU usage.",
                                    error_code="reading_unavailable")
    cores = f" across {health.cpu_cores} cores" if health.cpu_cores else ""
    return ActionResult.success(f"CPU is at {health.cpu_percent:.0f}%{cores}.",
                                cpu_percent=health.cpu_percent,
                                cpu_cores=health.cpu_cores)


def _memory_usage(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    if health.memory_percent is None:
        return ActionResult.failure("I couldn't read the memory usage.",
                                    error_code="reading_unavailable")
    used, total = health.memory_used_gb, health.memory_total_gb
    detail = (f" -- {used:.1f} of {total:.1f} GB"
              if used is not None and total is not None else "")
    return ActionResult.success(
        f"Memory is at {health.memory_percent:.0f}%{detail}.",
        memory_percent=health.memory_percent,
        memory_used_gb=used, memory_total_gb=total)


def _disk_usage(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    if not health.disks:
        return ActionResult.failure("I couldn't read any drives.",
                                    error_code="reading_unavailable")
    lines = [f"{d.mount} has {d.free_gb:.0f} GB free of {d.total_gb:.0f} GB "
             f"({d.percent:.0f}% used)" for d in health.disks]
    return ActionResult.success("; ".join(lines) + ".",
                                disks=[d.as_dict() for d in health.disks])


def _battery_status(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    if health.battery_percent is None:
        # A desktop has no battery. That is not a failure to measure.
        return ActionResult.success(
            "This machine doesn't report a battery.", battery_percent=None)
    state = ("charging" if health.battery_charging else "on battery"
             if health.battery_charging is False else "")
    left = ""
    if health.battery_minutes_left:
        hours, minutes = divmod(health.battery_minutes_left, 60)
        left = (f", about {hours}h {minutes}m left" if hours
                else f", about {minutes} minutes left")
    return ActionResult.success(
        f"Battery is at {health.battery_percent:.0f}%"
        + (f" and {state}" if state else "") + left + ".",
        battery_percent=health.battery_percent,
        battery_charging=health.battery_charging,
        battery_minutes_left=health.battery_minutes_left)


def _temperature(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Answer honestly, which on most Windows laptops means "I can't tell"."""
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    if health.cpu_temperature_c is None:
        reason = health.unavailable.get("cpu_temperature", "")
        return ActionResult.failure(
            "I can't read this machine's temperature -- "
            + (reason or "no sensor is exposed to me")
            + ". I'd rather say so than guess.",
            error_code="reading_unavailable")
    return ActionResult.success(
        f"CPU temperature is {health.cpu_temperature_c:.0f}°C.",
        cpu_temperature_c=health.cpu_temperature_c)


def _system_info(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    health = provider.health()
    bits: List[str] = []
    if health.os_name:
        bits.append(f"{health.os_name} {health.os_version}".strip())
    if health.cpu_cores:
        bits.append(f"{health.cpu_cores} logical cores")
    if health.memory_total_gb is not None:
        bits.append(f"{health.memory_total_gb:.1f} GB of memory")
    bits.append(f"up for {health.uptime_text()}")
    if health.process_count is not None:
        bits.append(f"{health.process_count} processes running")
    return ActionResult.success("This machine: " + ", ".join(bits) + ".",
                                **health.as_dict())


def _top_processes(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    by = str(params.get("resource") or "cpu").lower()
    by = "memory" if by.startswith("mem") or by == "ram" else "cpu"
    limit = int(params.get("count") or 5)
    processes = provider.top_processes(by=by, limit=limit)
    if not processes:
        return ActionResult.failure("I couldn't read the running processes.",
                                    error_code="reading_unavailable")
    if by == "memory":
        listed = ", ".join(f"{p.name} ({p.memory_mb:.0f} MB)"
                           for p in processes if p.memory_mb is not None)
        label = "memory"
    else:
        listed = ", ".join(f"{p.name} ({p.cpu_percent:.0f}%)"
                           for p in processes if p.cpu_percent is not None)
        label = "CPU"
    if not listed:
        return ActionResult.failure(f"I couldn't read {label} usage per process.",
                                    error_code="reading_unavailable")
    return ActionResult.success(f"Using the most {label}: {listed}.",
                                resource=by,
                                processes=[p.as_dict() for p in processes])


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="system_health", description="Report how the machine is doing.",
            category="System", risk=RiskLevel.LOW, handler=_system_health,
            examples=("check system health", "how is my PC doing",
                      "how's my computer"),
        ),
        ActionSpec(
            name="cpu_usage", description="Report current CPU usage.",
            category="System", risk=RiskLevel.LOW, handler=_cpu_usage,
            examples=("how much CPU am I using", "show my CPU usage"),
        ),
        ActionSpec(
            name="memory_usage", description="Report current memory usage.",
            category="System", risk=RiskLevel.LOW, handler=_memory_usage,
            examples=("how much RAM am I using", "check my memory"),
        ),
        ActionSpec(
            name="disk_usage", description="Report free disk space.",
            category="System", risk=RiskLevel.LOW, handler=_disk_usage,
            examples=("how much disk space do I have", "check my storage"),
        ),
        ActionSpec(
            name="battery_status", description="Report battery level and charging state.",
            category="System", risk=RiskLevel.LOW, handler=_battery_status,
            examples=("what's my battery level", "am I charging"),
        ),
        ActionSpec(
            name="cpu_temperature",
            description="Report CPU temperature, when a sensor is available.",
            category="System", risk=RiskLevel.LOW, handler=_temperature,
            examples=("is my laptop overheating", "what's my CPU temperature"),
        ),
        ActionSpec(
            name="system_info", description="Report the machine's specification.",
            category="System", risk=RiskLevel.LOW, handler=_system_info,
            examples=("show system information", "what are my specs"),
        ),
        ActionSpec(
            name="top_processes",
            description="Show which programs are using the most CPU or memory.",
            category="System", risk=RiskLevel.LOW, handler=_top_processes,
            parameters=(
                ParameterSpec(name="resource", description="cpu or memory",
                              required=False, default="cpu",
                              choices=("cpu", "memory")),
                ParameterSpec(name="count", description="how many to list",
                              required=False, default=5),
            ),
            examples=("what is using the most CPU", "what's using my memory"),
        ),
    ]

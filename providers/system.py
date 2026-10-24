"""What the machine is actually doing: CPU, memory, disk, battery, uptime.

Everything here is measured. Where a reading is not obtainable -- and on a
laptop the commonest one is CPU temperature, which Windows does not expose
through any API ``psutil`` can reach without a vendor driver -- the field is
``None`` and the assistant says it does not know.

That distinction is the whole point of this module. "Your CPU is at 65°C" when
nothing measured 65°C is a fabrication the user has no way to catch, and it is
exactly the kind of claim a voice assistant makes most convincingly.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from providers import Availability, Provider
from utils.logger import get_logger

logger = get_logger("providers.system")

#: How long to wait between the two samples a per-process CPU reading needs.
#: Short, because someone asked out loud and is waiting for the answer.
CPU_SAMPLE_SECONDS = 0.4


@dataclass(frozen=True)
class DiskUsage:
    """One mounted drive."""

    mount: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def percent(self) -> float:
        return (self.used_bytes / self.total_bytes * 100.0) if self.total_bytes else 0.0

    @property
    def free_gb(self) -> float:
        return self.free_bytes / 1_000_000_000

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1_000_000_000

    def as_dict(self) -> Dict[str, Any]:
        return {"mount": self.mount, "total_gb": round(self.total_gb, 1),
                "free_gb": round(self.free_gb, 1), "percent": round(self.percent, 1)}


@dataclass(frozen=True)
class SystemHealth:
    """A snapshot of the machine.

    Every optional field is ``None`` when the reading could not be taken. Nothing
    here is estimated, interpolated or defaulted to a plausible value.
    """

    cpu_percent: Optional[float] = None
    cpu_cores: Optional[int] = None
    cpu_temperature_c: Optional[float] = None
    memory_total_bytes: Optional[int] = None
    memory_used_bytes: Optional[int] = None
    memory_percent: Optional[float] = None
    disks: List[DiskUsage] = field(default_factory=list)
    battery_percent: Optional[float] = None
    battery_charging: Optional[bool] = None
    battery_minutes_left: Optional[int] = None
    uptime_seconds: Optional[float] = None
    process_count: Optional[int] = None
    os_name: str = ""
    os_version: str = ""
    network_up: Optional[bool] = None
    measured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    #: Readings that were attempted and could not be taken, with the reason.
    unavailable: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- derived

    @property
    def memory_used_gb(self) -> Optional[float]:
        if self.memory_used_bytes is None:
            return None
        return self.memory_used_bytes / 1_000_000_000

    @property
    def memory_total_gb(self) -> Optional[float]:
        if self.memory_total_bytes is None:
            return None
        return self.memory_total_bytes / 1_000_000_000

    @property
    def primary_disk(self) -> Optional[DiskUsage]:
        return self.disks[0] if self.disks else None

    def uptime_text(self) -> str:
        if self.uptime_seconds is None:
            return "unknown"
        delta = timedelta(seconds=int(self.uptime_seconds))
        days, rest = delta.days, delta.seconds
        hours, minutes = rest // 3600, (rest % 3600) // 60
        if days:
            return f"{days}d {hours}h {minutes}m"
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    # -------------------------------------------------------------- speech

    def summary(self) -> str:
        """One sentence a person would actually say, omitting what is unknown."""
        parts: List[str] = []
        if self.cpu_percent is not None:
            parts.append(f"CPU is at {self.cpu_percent:.0f}%")
        if self.memory_percent is not None:
            used, total = self.memory_used_gb, self.memory_total_gb
            if used is not None and total is not None:
                parts.append(f"memory at {self.memory_percent:.0f}% "
                             f"({used:.1f} of {total:.1f} GB)")
            else:
                parts.append(f"memory at {self.memory_percent:.0f}%")
        if self.battery_percent is not None:
            state = ("charging" if self.battery_charging else "on battery"
                     if self.battery_charging is False else "")
            parts.append(f"battery is {self.battery_percent:.0f}%"
                         + (f" and {state}" if state else ""))
        disk = self.primary_disk
        if disk is not None:
            parts.append(f"{disk.mount} has {disk.free_gb:.0f} GB free")
        if not parts:
            return "I couldn't read anything about this machine."
        def lead(text: str) -> str:
            # Only the first character: .capitalize() would turn "CPU" into
            # "Cpu" and "14.8 GB" into "14.8 gb".
            return text[:1].upper() + text[1:] if text else text

        if len(parts) == 1:
            return lead(parts[0]) + "."
        return lead(", ".join(parts[:-1])) + f", and {parts[-1]}."

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent, "cpu_cores": self.cpu_cores,
            "cpu_temperature_c": self.cpu_temperature_c,
            "memory_percent": self.memory_percent,
            "memory_used_gb": (round(self.memory_used_gb, 2)
                               if self.memory_used_gb is not None else None),
            "memory_total_gb": (round(self.memory_total_gb, 2)
                                if self.memory_total_gb is not None else None),
            "disks": [d.as_dict() for d in self.disks],
            "battery_percent": self.battery_percent,
            "battery_charging": self.battery_charging,
            "battery_minutes_left": self.battery_minutes_left,
            "uptime": self.uptime_text(), "process_count": self.process_count,
            "os": f"{self.os_name} {self.os_version}".strip(),
            "network_up": self.network_up,
            "measured_at": self.measured_at.isoformat(timespec="seconds"),
            "unavailable": dict(self.unavailable),
        }


@dataclass(frozen=True)
class ProcessInfo:
    """One running process, as much as could be read without elevation."""

    pid: int
    name: str
    cpu_percent: Optional[float] = None
    memory_bytes: Optional[int] = None

    @property
    def memory_mb(self) -> Optional[float]:
        return self.memory_bytes / 1_048_576 if self.memory_bytes is not None else None

    def as_dict(self) -> Dict[str, Any]:
        return {"pid": self.pid, "name": self.name,
                "cpu_percent": self.cpu_percent,
                "memory_mb": (round(self.memory_mb, 1)
                              if self.memory_mb is not None else None)}


class SystemProvider(Provider):
    """Reads the machine through ``psutil``, reporting gaps rather than filling them."""

    name = "system"
    description = "CPU, memory, disk, battery and running processes"

    def __init__(self, psutil_module: Any = None) -> None:
        # Injectable so the tests can drive a machine that has no battery, or a
        # psutil that raises, without needing that machine.
        self._psutil = psutil_module
        self._resolved = psutil_module is not None

    # ------------------------------------------------------------ plumbing

    def _module(self) -> Any:
        if not self._resolved:
            try:
                import psutil  # noqa: PLC0415 - optional, resolved once

                self._psutil = psutil
            except Exception as exc:  # noqa: BLE001
                logger.warning("psutil unavailable", extra={
                    "context": {"error": str(exc)}})
                self._psutil = None
            self._resolved = True
        return self._psutil

    def availability(self) -> Availability:
        if self._module() is None:
            return Availability.missing(
                self.name, "I can't read this machine's vitals.",
                "The psutil package isn't installed.")
        return Availability.ready(self.name, "CPU, memory, disk and battery")

    # -------------------------------------------------------------- health

    def health(self, *, cpu_sample_seconds: float = 0.3) -> SystemHealth:
        """Take a snapshot.  Never raises; unreadable fields stay ``None``.

        ``cpu_sample_seconds`` is short on purpose: a health check is something
        the user asked for out loud and is waiting on, and psutil's instant
        reading is meaningless without an interval.
        """
        psutil = self._module()
        if psutil is None:
            return SystemHealth(
                os_name=platform.system(), os_version=platform.release(),
                unavailable={"all": "psutil is not installed"})

        missing: Dict[str, str] = {}

        def attempt(label: str, fn):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - one gap must not lose the rest
                logger.debug("System reading failed: %s (%s)", label, exc)
                missing[label] = f"{type(exc).__name__}: {exc}"
                return None

        cpu = attempt("cpu", lambda: float(psutil.cpu_percent(cpu_sample_seconds)))
        cores = attempt("cpu_cores", lambda: int(psutil.cpu_count(logical=True) or 0))

        memory = attempt("memory", psutil.virtual_memory)
        disks = attempt("disks", lambda: self._disks(psutil)) or []
        battery = attempt("battery", lambda: self._battery(psutil))
        boot = attempt("uptime", lambda: float(psutil.boot_time()))
        processes = attempt("processes", lambda: len(psutil.pids()))
        temperature, temp_reason = self._temperature(psutil)
        if temperature is None and temp_reason:
            missing["cpu_temperature"] = temp_reason

        return SystemHealth(
            cpu_percent=cpu, cpu_cores=cores, cpu_temperature_c=temperature,
            memory_total_bytes=getattr(memory, "total", None),
            memory_used_bytes=getattr(memory, "used", None),
            memory_percent=getattr(memory, "percent", None),
            disks=disks,
            battery_percent=battery[0] if battery else None,
            battery_charging=battery[1] if battery else None,
            battery_minutes_left=battery[2] if battery else None,
            uptime_seconds=(time.time() - boot) if boot else None,
            process_count=processes,
            os_name=platform.system(), os_version=platform.release(),
            network_up=self._network(psutil),
            unavailable=missing,
        )

    @staticmethod
    def _disks(psutil: Any) -> List[DiskUsage]:
        found: List[DiskUsage] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except Exception:  # noqa: BLE001 - an unreadable drive is skipped
                continue
            found.append(DiskUsage(mount=part.mountpoint, total_bytes=usage.total,
                                   used_bytes=usage.used, free_bytes=usage.free))
        return found

    @staticmethod
    def _battery(psutil: Any):
        getter = getattr(psutil, "sensors_battery", None)
        if getter is None:
            return None
        battery = getter()
        if battery is None:              # a desktop: not a failure, just no battery
            return None
        minutes = None
        secs = getattr(battery, "secsleft", None)
        if isinstance(secs, (int, float)) and secs >= 0:
            minutes = int(secs // 60)
        return (float(battery.percent), bool(battery.power_plugged), minutes)

    @staticmethod
    def _temperature(psutil: Any):
        """CPU temperature, or the reason there isn't one.

        ``sensors_temperatures`` does not exist on Windows at all, and where it
        does exist it can return an empty mapping. Both mean the same thing to
        the user: no reading. Neither means "it is fine".
        """
        getter = getattr(psutil, "sensors_temperatures", None)
        if getter is None:
            return None, "this platform exposes no temperature sensor to psutil"
        try:
            readings = getter()
        except Exception as exc:  # noqa: BLE001
            return None, f"the temperature sensor could not be read ({exc})"
        if not readings:
            return None, "no temperature sensor reported a value"
        for entries in readings.values():
            for entry in entries:
                current = getattr(entry, "current", None)
                if isinstance(current, (int, float)):
                    return float(current), ""
        return None, "the temperature sensors reported no usable value"

    @staticmethod
    def _network(psutil: Any) -> Optional[bool]:
        try:
            stats = psutil.net_if_stats()
        except Exception:  # noqa: BLE001
            return None
        return any(getattr(s, "isup", False) for name, s in stats.items()
                   if "loopback" not in name.lower())

    # ----------------------------------------------------------- processes

    def top_processes(self, *, by: str = "cpu", limit: int = 5) -> List[ProcessInfo]:
        """The heaviest processes.  ``by`` is ``"cpu"`` or ``"memory"``.

        Only the name, pid and usage are read. Command lines are deliberately
        not collected: they routinely contain tokens and passwords, and nothing
        here needs them.
        """
        psutil = self._module()
        if psutil is None:
            return []
        found: List[ProcessInfo] = []
        try:
            if by != "memory":
                # psutil measures process CPU *between* two calls; the first
                # always returns 0.0.  Without this prime-and-wait every
                # process ties at zero and the answer is meaningless.
                for proc in psutil.process_iter():
                    try:
                        proc.cpu_percent(None)
                    except Exception:  # noqa: BLE001 - a vanished process
                        continue
                time.sleep(CPU_SAMPLE_SECONDS)
            for proc in psutil.process_iter(["pid", "name", "cpu_percent",
                                             "memory_info"]):
                info = proc.info
                memory = info.get("memory_info")
                found.append(ProcessInfo(
                    pid=int(info.get("pid") or 0),
                    name=str(info.get("name") or "?"),
                    cpu_percent=info.get("cpu_percent"),
                    memory_bytes=getattr(memory, "rss", None)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Process listing failed", extra={
                "context": {"error": str(exc)}})
            return []

        if by == "memory":
            found.sort(key=lambda p: p.memory_bytes or 0, reverse=True)
        else:
            found.sort(key=lambda p: p.cpu_percent or 0.0, reverse=True)
        return found[:max(1, limit)]


#: The process-wide provider.
SYSTEM = SystemProvider()

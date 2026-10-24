from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .config import SerialCfg

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

PORT_HINTS = ("ttyACM", "ttyUSB", "usbmodem", "usbserial", "COM")


@dataclass
class LinkStatus:
    connected: bool = False
    port: str = ""
    sends: int = 0
    errors: int = 0
    last_line: str = ""


def find_port(cfg: SerialCfg) -> str | None:
    if cfg.port and cfg.port != "auto":
        return cfg.port
    if list_ports is None:
        return None
    candidates = []
    for info in list_ports.comports():
        name = info.device or ""
        if any(hint.lower() in name.lower() for hint in PORT_HINTS):
            candidates.append(name)
    return sorted(candidates)[0] if candidates else None


class NullLink:
    def __init__(self) -> None:
        self.status = LinkStatus(connected=False, port="none", last_line="sin arduino")

    def open(self) -> bool:
        return False

    def send(self, left: int, right: int) -> None:
        self.status.sends += 1

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class SerialLink:
    def __init__(self, cfg: SerialCfg) -> None:
        self.cfg = cfg
        self.status = LinkStatus()
        self._port = None
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._running = False
        self._next_retry = 0.0

    def open(self) -> bool:
        if serial is None:
            self.status.last_line = "pyserial no instalado"
            return False
        port = find_port(self.cfg)
        if port is None:
            self.status.last_line = "no encontre puerto"
            self._next_retry = time.monotonic() + 3.0
            return False
        try:
            self._port = serial.Serial(port, self.cfg.baud, timeout=0, write_timeout=0.2)
        except Exception as exc:
            self.status.last_line = f"abrir {port}: {exc.__class__.__name__}"
            self._next_retry = time.monotonic() + 3.0
            return False

        self.status.connected = True
        self.status.port = port
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, name="iris-serial", daemon=True)
        self._reader.start()
        time.sleep(0.4)
        return True

    def _read_loop(self) -> None:
        while self._running and self._port is not None:
            try:
                chunk = self._port.readline()
            except Exception:
                self.status.connected = False
                self.status.errors += 1
                return
            if not chunk:
                time.sleep(0.02)
                continue
            try:
                line = chunk.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not line:
                continue
            self.status.last_line = line

    def send(self, left: int, right: int) -> None:
        if self._port is None or not self.status.connected:
            if time.monotonic() >= self._next_retry:
                self.open()
            return
        payload = f"D {left} {right}\n".encode("ascii")
        try:
            with self._lock:
                self._port.write(payload)
            self.status.sends += 1
        except Exception as exc:
            self.status.connected = False
            self.status.errors += 1
            self.status.last_line = f"write: {exc.__class__.__name__}"

    def stop(self) -> None:
        self.send(0, 0)

    def close(self) -> None:
        self._running = False
        try:
            if self._port is not None:
                with self._lock:
                    self._port.write(b"S\n")
                    self._port.flush()
        except Exception:
            pass
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass
        self._port = None
        self.status.connected = False


def build_link(cfg: SerialCfg):
    if not cfg.enabled:
        return NullLink()
    return SerialLink(cfg)

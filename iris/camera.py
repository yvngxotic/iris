from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from .config import CameraCfg


@dataclass
class CameraStats:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    brightness: float = 0.0


class Camera:
    def __init__(self, cfg: CameraCfg) -> None:
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None
        self.failures = 0

    def _source(self):
        src = self.cfg.source
        if isinstance(src, str) and src.strip().isdigit():
            return int(src.strip())
        return src

    def open(self) -> bool:
        source = self._source()
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.cap = None
            return False

        if isinstance(source, int) and self.cfg.fourcc:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.cfg.fourcc[:4].upper().ljust(4, " ")),
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if isinstance(source, int):
            self._apply_locks()
        for _ in range(max(0, self.cfg.warmup_frames)):
            self.cap.read()
        return True

    def _apply_locks(self) -> None:
        if self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if self.cfg.auto_wb else 0)
        except Exception:
            pass
        if self.cfg.lock_exposure:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.exposure)
            except Exception:
                pass

    def read(self):
        if self.cap is None:
            if not self.open():
                return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures >= 30:
                self.close()
                self.failures = 0
                time.sleep(0.3)
            return None
        self.failures = 0
        return frame

    def medir(self, frames: int = 24) -> CameraStats:
        if self.cap is None and not self.open():
            return CameraStats()
        for _ in range(4):
            self.cap.read()
        started = time.perf_counter()
        frame = None
        measured = 0
        for _ in range(frames):
            ok, current = self.cap.read()
            if not ok or current is None:
                break
            frame = current
            measured += 1
        elapsed = time.perf_counter() - started
        stats = CameraStats(fps=measured / elapsed if elapsed > 0 and measured else 0.0)
        if frame is not None:
            stats.height, stats.width = frame.shape[:2]
            stats.brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        stats.fourcc = code.to_bytes(4, "little").decode("ascii", errors="replace")
        return stats

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

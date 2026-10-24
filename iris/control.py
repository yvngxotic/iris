from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ControlCfg
from .vision import CarPose

STOP = "SIN VEHICULO"
NO_PATH = "SIN RUTA"
BLIND = "CALIBRANDO"
BLOCKED = "OBSTACULO"
FOLLOW = "SIGUIENDO"


@dataclass
class Drive:
    left: int = 0
    right: int = 0
    state: str = STOP
    error: float = 0.0
    clearance: float = 0.0
    target: tuple[float, float] | None = None

    def scaled(self, factor: float) -> Drive:
        return Drive(
            left=int(round(self.left * factor)),
            right=int(round(self.right * factor)),
            state=self.state,
            error=self.error,
            clearance=self.clearance,
            target=self.target,
        )


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _apply_deadband(value: int, min_pwm: int) -> int:
    if value == 0:
        return 0
    if abs(value) < min_pwm:
        return int(math.copysign(min_pwm, value))
    return value


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ahead(pose: CarPose, point: tuple[float, float], margin: float = 2.0) -> bool:
    dx = point[0] - pose.x
    dy = point[1] - pose.y
    return (dx * math.cos(pose.heading) + dy * math.sin(pose.heading)) > margin


def _segment_clear(
    mask, start: tuple[float, float], end: tuple[float, float], step: float = 3.0
) -> bool:
    span = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(2, int(span / step))
    height, width = mask.shape[:2]
    for index in range(steps + 1):
        blend = index / steps
        px = int(round(start[0] + (end[0] - start[0]) * blend))
        py = int(round(start[1] + (end[1] - start[1]) * blend))
        if px < 0 or py < 0 or px >= width or py >= height:
            return False
        if mask[py, px] > 0:
            return False
    return True


class PurePursuit:
    def __init__(self, cfg: ControlCfg) -> None:
        self.cfg = cfg
        self.turn_gain = 120.0 * cfg.heading_gain

    def _target(
        self,
        pose: CarPose,
        path_px: list[tuple[float, float]],
        lookahead: float,
        mask=None,
    ) -> tuple[float, float]:
        nearest = 0
        best = 1e18
        for idx, (px, py) in enumerate(path_px):
            dist = (px - pose.x) ** 2 + (py - pose.y) ** 2
            if dist < best:
                best, nearest = dist, idx

        travelled = 0.0
        prev = (pose.x, pose.y)
        reach = len(path_px) - 1
        for idx in range(nearest, len(path_px)):
            point = path_px[idx]
            travelled += math.hypot(point[0] - prev[0], point[1] - prev[1])
            if travelled >= lookahead:
                reach = idx
                break
            prev = point

        if mask is not None:
            for idx in range(reach, nearest - 1, -1):
                point = path_px[idx]
                if _ahead(pose, point) and _segment_clear(mask, (pose.x, pose.y), point):
                    return point
        return path_px[reach]

    def update(
        self,
        pose: CarPose,
        path_px: list[tuple[float, float]],
        clearance: float,
        obstacle_mask=None,
    ) -> Drive:
        if not pose.visible:
            return Drive(left=0, right=0, state=STOP)
        if len(path_px) < 2:
            return Drive(left=0, right=0, state=NO_PATH, clearance=clearance)

        if not pose.confident:
            speed = int(self.cfg.max_pwm * 0.4)
            return Drive(left=speed, right=speed, state=BLIND, clearance=clearance)

        lookahead = _clip(clearance * 0.55, 16.0, self.cfg.lookahead_px)
        target = self._target(pose, path_px, lookahead, obstacle_mask)
        error = _wrap(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.heading)

        if clearance <= self.cfg.stop_distance_px:
            return Drive(
                left=0,
                right=0,
                state=BLOCKED,
                error=error,
                clearance=clearance,
                target=target,
            )

        if not pose.verified and abs(error) > 2.2:
            speed = int(self.cfg.max_pwm * 0.5)
            return Drive(
                left=speed,
                right=speed,
                state=BLIND,
                error=error,
                clearance=clearance,
                target=target,
            )

        margin = self.cfg.stop_distance_px
        room = _clip((clearance - margin) / (margin * 3.0), 0.15, 1.0)
        slowdown = room * (1.0 - self.cfg.turn_slowdown * min(1.0, abs(error) / (math.pi / 2.0)))
        speed = self.cfg.max_pwm * slowdown
        turn = self.turn_gain * error

        limit = self.cfg.max_pwm
        left = _clip(speed - turn, -limit, limit)
        right = _clip(speed + turn, -limit, limit)
        return Drive(
            left=_apply_deadband(int(round(left)), self.cfg.min_pwm),
            right=_apply_deadband(int(round(right)), self.cfg.min_pwm),
            state=FOLLOW,
            error=error,
            clearance=clearance,
            target=target,
        )

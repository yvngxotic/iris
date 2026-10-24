from __future__ import annotations

import math

import cv2
import numpy as np

from .config import SimCfg

BLUE = (255, 60, 30)
WHITE = (238, 238, 238)
WALL = (112, 112, 112)
PAD = (22, 22, 22)
PAD_ALT = (48, 48, 48)


def _quads_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for polygon in (a, b):
        for index in range(len(polygon)):
            first = polygon[index]
            second = polygon[(index + 1) % len(polygon)]
            axis = (-(second[1] - first[1]), second[0] - first[0])
            proj_a = a @ np.array(axis)
            proj_b = b @ np.array(axis)
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


class SimWorld:
    def __init__(self, cfg: SimCfg) -> None:
        self.cfg = cfg
        self.w = cfg.arena_w
        self.h = cfg.arena_h
        self.car = [self.w * 0.16, self.h * 0.5, 0.0]
        self.command = (0, 0)
        self.collisions = 0
        self._collision_cooldown = 0.0
        self.obstacles = self._layout()
        self._noise = np.random.randint(0, 7, (self.h, self.w, 3), dtype=np.uint8)

    def _layout(self) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
        w, h = self.w, self.h
        wall = 12
        return [
            (0, 0, w, wall, WALL),
            (0, h - wall, w, wall, WALL),
            (0, 0, wall, h, WALL),
            (w - wall, 0, wall, h, WALL),
            (int(w * 0.24), int(h * 0.04), int(w * 0.07), int(h * 0.36), PAD),
            (int(w * 0.24), int(h * 0.62), int(w * 0.07), int(h * 0.34), PAD),
            (int(w * 0.45), int(h * 0.36), int(w * 0.11), int(h * 0.28), PAD_ALT),
            (
                int(w * 0.69),
                int(h * 0.04),
                int(w * 0.07),
                int(h * 0.32),
                PAD,
            ),
            (
                int(w * 0.69),
                int(h * 0.64),
                int(w * 0.07),
                int(h * 0.32),
                PAD,
            ),
        ]

    @property
    def goal_px(self) -> tuple[float, float]:
        return self.w * 0.88, self.h * 0.5

    def corners(
        self, x: float | None = None, y: float | None = None, theta: float | None = None
    ) -> np.ndarray:
        cx = self.car[0] if x is None else x
        cy = self.car[1] if y is None else y
        th = self.car[2] if theta is None else theta
        half_l = self.cfg.car_len / 2.0
        half_w = self.cfg.car_wid / 2.0
        local = np.array(
            [[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]]
        )
        cos_t, sin_t = math.cos(th), math.sin(th)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return (local @ rot.T) + np.array([cx, cy])

    def _blocked(self, corners: np.ndarray) -> bool:
        for x, y, w, h, _ in self.obstacles:
            pad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)
            if _quads_overlap(corners, pad):
                return True
        return False

    def set_command(self, left: int, right: int) -> None:
        self.command = (int(left), int(right))

    def step(self, dt: float) -> None:
        left, right = self.command
        v = (left + right) / 2.0 / 255.0 * self.cfg.max_speed_px_s
        omega = (right - left) / 2.0 / 255.0 * self.cfg.max_turn_rad_s
        dt = min(dt, 0.1)
        new_theta = self.car[2] + omega * dt
        new_x = self.car[0] + math.cos(new_theta) * v * dt
        new_y = self.car[1] + math.sin(new_theta) * v * dt
        if self._blocked(self.corners(new_x, new_y, new_theta)):
            if self._collision_cooldown <= 0.0:
                self.collisions += 1
                self._collision_cooldown = 0.5
            self.command = (0, 0)
        else:
            self.car = [new_x, new_y, new_theta]
        self._collision_cooldown = max(0.0, self._collision_cooldown - dt)

    def render(self) -> np.ndarray:
        frame = np.full((self.h, self.w, 3), WHITE, dtype=np.uint8)
        for x, y, w, h, color in self.obstacles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        box = self.corners().astype(np.int32)
        cv2.fillPoly(frame, [box], BLUE)
        return cv2.add(frame, self._noise)


class SimLink:
    def __init__(self, world: SimWorld) -> None:
        from .serial_link import LinkStatus

        self.world = world
        self.status = LinkStatus(connected=True, port="sim")
        self.status.last_line = "simulado"

    def open(self) -> bool:
        return True

    def send(self, left: int, right: int) -> None:
        self.world.set_command(left, right)
        self.status.sends += 1

    def stop(self) -> None:
        self.world.set_command(0, 0)

    def close(self) -> None:
        self.world.set_command(0, 0)


def default_goal(cfg: SimCfg) -> tuple[int, int]:
    return int(cfg.arena_w * 0.88), int(cfg.arena_h * 0.5)

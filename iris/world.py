from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import WorldCfg
from .vision import Scene


@dataclass
class Grid:
    free: np.ndarray
    blocked: np.ndarray
    cell_px: float
    car_cell: tuple[int, int] | None
    raw_blocked: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.blocked.shape

    @property
    def rows(self) -> int:
        return int(self.blocked.shape[0])

    @property
    def cols(self) -> int:
        return int(self.blocked.shape[1])

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return 0 <= col < self.cols and 0 <= row < self.rows

    def walkable(self, cell: tuple[int, int]) -> bool:
        if not self.in_bounds(cell):
            return False
        return bool(self.free[cell[1], cell[0]])

    def to_px(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (cell[0] + 0.5) * self.cell_px, (cell[1] + 0.5) * self.cell_px

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.cell_px), int(y // self.cell_px)


def _inflate(blocked: np.ndarray, cells: int) -> np.ndarray:
    if cells <= 0:
        return blocked
    out = blocked.copy()
    for _ in range(cells):
        padded = np.pad(out, 1, constant_values=True)
        grown = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = grown
    return out


def armar_grid(scene: Scene, cfg: WorldCfg, obstacle_mask=None) -> Grid:
    proc = scene.proc
    cell = float(cfg.cell_px)
    cols = max(4, int(proc.shape[1] // cell))
    rows = max(4, int(proc.shape[0] // cell))
    size = (cols, rows)

    obs_frac = (
        cv2.resize(
            scene.obstacle_mask if obstacle_mask is None else obstacle_mask,
            size,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        / 255.0
    )
    floor_frac = (
        cv2.resize(scene.floor_mask, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    )

    free = (floor_frac >= cfg.floor_frac_min) & (obs_frac < cfg.obstacle_frac)

    car_cell = None
    if scene.car.visible:
        car_cell = (int(scene.car.x // cell), int(scene.car.y // cell))
        reach = int(math.ceil(scene.car.radius / cell)) + 1
        col0, row0 = car_cell
        yy, xx = np.ogrid[:rows, :cols]
        under_car = ((xx - col0) ** 2 + (yy - row0) ** 2) <= reach * reach
        free = free | (under_car & (obs_frac < cfg.obstacle_frac))

    blocked = ~free
    raw_blocked = blocked.copy()
    inflate_cells = 0
    if scene.car.visible:
        plan_radius = scene.car.half_wid + cell * 0.5
        inflate_cells = int(math.ceil(plan_radius / cell)) + int(cfg.safety_cells)
    blocked = _inflate(blocked, inflate_cells)

    if car_cell is not None and 0 <= car_cell[1] < rows and 0 <= car_cell[0] < cols:
        blocked[car_cell[1], car_cell[0]] = False
        raw_blocked[car_cell[1], car_cell[0]] = False

    return Grid(
        free=~blocked,
        blocked=blocked,
        cell_px=cell,
        car_cell=car_cell,
        raw_blocked=raw_blocked,
    )


def render_ascii(
    grid: Grid,
    car: tuple[int, int] | None,
    goal: tuple[int, int] | None,
    path: list[tuple[int, int]] | None,
    width: int = 40,
) -> str:
    rows, cols = grid.shape
    out_rows = max(8, int(round(rows * width / cols / 2.0)))
    lines: list[str] = []
    path_set = set(path or [])
    for out_r in range(out_rows):
        row0 = int(out_r * rows / out_rows)
        row1 = max(row0 + 1, int((out_r + 1) * rows / out_rows))
        chars = []
        for out_c in range(width):
            col0 = int(out_c * cols / width)
            col1 = max(col0 + 1, int((out_c + 1) * cols / width))
            block = range(col0, col1), range(row0, row1)
            if car and col0 <= car[0] < col1 and row0 <= car[1] < row1:
                chars.append("C")
            elif goal and col0 <= goal[0] < col1 and row0 <= goal[1] < row1:
                chars.append("G")
            elif any((c, r) in path_set for c in block[0] for r in block[1]):
                chars.append("o")
            elif grid.blocked[row0:row1, col0:col1].mean() >= 0.5:
                chars.append("#")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)

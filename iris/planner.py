from __future__ import annotations

import heapq
import math

from .world import Grid

NEIGHBORS = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
)


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)


def nearest_walkable(
    grid: Grid, cell: tuple[int, int], max_radius: int = 12
) -> tuple[int, int] | None:
    if grid.walkable(cell):
        return cell
    col0, row0 = cell
    for radius in range(1, max_radius + 1):
        best = None
        best_d = 1e9
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cand = (col0 + dx, row0 + dy)
                if not grid.walkable(cand):
                    continue
                dist = dx * dx + dy * dy
                if dist < best_d:
                    best, best_d = cand, dist
        if best is not None:
            return best
    return None


def plan(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_expansions: int = 20000,
) -> list[tuple[int, int]] | None:
    start_cell = nearest_walkable(grid, start)
    goal_cell = nearest_walkable(grid, goal)
    if start_cell is None or goal_cell is None:
        return None
    if start_cell == goal_cell:
        return [start_cell]

    cost: dict[tuple[int, int], float] = {start_cell: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_cell)]
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while heap:
        _, g_cur, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if expansions > max_expansions:
            return None
        if current == goal_cell:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy, step in NEIGHBORS:
            nxt = (current[0] + dx, current[1] + dy)
            if not grid.walkable(nxt) or nxt in closed:
                continue
            if dx and dy:
                if not grid.walkable((current[0] + dx, current[1])) or not grid.walkable(
                    (current[0], current[1] + dy)
                ):
                    continue
            g_new = g_cur + step
            if g_new < cost.get(nxt, 1e18):
                cost[nxt] = g_new
                parent[nxt] = current
                heapq.heappush(heap, (g_new + _octile(nxt, goal_cell), g_new, nxt))
    return None


def line_of_sight(grid: Grid, a: tuple[int, int], b: tuple[int, int]) -> bool:
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy)
    if steps == 0:
        return True
    for i in range(1, steps + 1):
        t = i / steps
        cell = (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
        if not grid.walkable(cell):
            return False
    return True


def smooth(grid: Grid, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    index = 0
    while index < len(path) - 1:
        farthest = index + 1
        for probe in range(len(path) - 1, index, -1):
            if line_of_sight(grid, path[index], path[probe]):
                farthest = probe
                break
        out.append(path[farthest])
        index = farthest
    return out


def resample(path: list[tuple[int, int]], step: float = 1.0) -> list[tuple[int, int]]:
    if len(path) < 2 or step <= 0:
        return list(path)
    out = [path[0]]
    for a, b in zip(path, path[1:], strict=False):
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        n = int(max(dx, dy) / step)
        if n <= 1:
            out.append(b)
            continue
        for i in range(1, n + 1):
            t = i / n
            cell = (
                int(round(a[0] + (b[0] - a[0]) * t)),
                int(round(a[1] + (b[1] - a[1]) * t)),
            )
            if cell != out[-1]:
                out.append(cell)
    return out


def plan_smooth(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_expansions: int = 20000,
) -> list[tuple[int, int]] | None:
    raw = plan(grid, start, goal, max_expansions=max_expansions)
    if not raw:
        return raw
    return resample(smooth(grid, raw), step=1.0)

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from .camera import Camera
from .config import Config
from .control import BLOCKED, FOLLOW, NO_PATH, STOP, Drive, PurePursuit
from .hud import Hud, HudData
from .llm import Decision, Strategist
from .planner import nearest_walkable, plan_smooth
from .serial_link import build_link
from .sim import SimLink, SimWorld
from .vision import Scene, Vision, holgura_frente
from .world import Grid, armar_grid, render_ascii

ESCAPE = "REVIRTIENDO"
PAUSED = "PAUSA"
AT_GOAL = "META"


@dataclass
class LoopStats:
    frames: int = 0
    fps: float = 0.0
    planner_ms: float = 0.0
    ai_calls: int = 0
    ai_errors: int = 0
    replans: int = 0
    stuck_events: int = 0
    goal_reached: bool = False
    min_goal_dist: float = 1e9


class App:
    def __init__(self, cfg: Config, args) -> None:
        self.cfg = cfg
        self.args = args
        self.sim = SimWorld(cfg.sim) if args.sim else None
        self.camera = None if args.sim else Camera(cfg.camera)
        self.vision = Vision(cfg.vision, cfg.camera.proc_width)
        self.pursuit = PurePursuit(cfg.control)
        self.link = SimLink(self.sim) if self.sim else build_link(cfg.serial)
        self.ai = Strategist(cfg.llm)
        self.grid: Grid | None = None
        self.scene: Scene | None = None
        self.path: list[tuple[int, int]] = []
        self.path_px: list[tuple[float, float]] = []
        self.drive = Drive()
        self.goal_px = self._initial_goal()
        self.ai_goal_cell: tuple[int, int] | None = None
        self.ai_text = "sin consultas todavia"
        self.speed_scale = 1.0
        self.paused = False
        self.estop = False
        self.at_goal = False
        self._escape_until = 0.0
        self._escape_dir = 1
        self._blocked_since: float | None = None
        self._plan_blocked_until = 0.0
        self._planned_at = -1e9
        self._planned_goal: tuple[int, int] | None = None
        self.clock = 0.0
        self.stats = LoopStats()
        self.events: deque[str] = deque(maxlen=6)
        self._stuck_ref: tuple[float, float, float] | None = None
        self.last_ai_at = 0.0
        self.frame_idx = 0
        self._fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=60)
        self._writer = None
        self._window = "iris"
        self._proc_shape: tuple[int, int] | None = None
        self.hud = Hud()

    def _initial_goal(self) -> tuple[float, float]:
        if self.sim is not None:
            scale = self.cfg.camera.proc_width / float(self.cfg.sim.arena_w)
            gx, gy = self.sim.goal_px
            return gx * scale, gy * scale
        if self.cfg.planner.goal:
            return float(self.cfg.planner.goal[0]), float(self.cfg.planner.goal[1])
        width = self.cfg.camera.proc_width
        return width * 0.85, width * 0.75 * 0.5

    def setup(self) -> bool:
        if self.sim is not None:
            print(f"[iris] simulacion {self.cfg.sim.arena_w}x{self.cfg.sim.arena_h}")
        else:
            if self.camera is None or not self.camera.open():
                print(f"[iris] no pude abrir la camara: {self.cfg.camera.source}")
                return False
        self.link.open()
        if self.sim is None:
            if self.link.status.connected:
                baud = self.cfg.serial.baud
                print(f"[iris] arduino en {self.link.status.port} ({baud} baud)")
            else:
                reason = self.link.status.last_line
                print(f"[iris] sin arduino ({reason}); el carro no se va a mover")
        if self.ai.available():
            self.ai.start()
            print(f"[iris] IA lista: {self.ai.model}")
        else:
            print("[iris] IA apagada o sin COMMANDCODE_API_KEY; control local solamente")
        if not self.args.headless:
            cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self._window, self._on_mouse)
        return True

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._proc_shape is None:
            return
        proc_w, proc_h = self._proc_shape
        if not (0 <= x < proc_w and 0 <= y < proc_h):
            return
        self.goal_px = (float(x), float(y))
        self.ai_goal_cell = None
        self.at_goal = False
        self.path = []
        self.events.append(f"meta nueva {x},{y}")

    def grab(self):
        if self.sim is not None:
            return self.sim.render()
        return self.camera.read()

    def step(self, frame, dt: float) -> None:
        self.clock += dt
        scene = self.vision.analizar(frame)
        self.scene = scene
        self._proc_shape = (scene.proc.shape[1], scene.proc.shape[0])
        estable = self.vision.suavizar_obstaculos(scene.obstacle_mask)
        grid = armar_grid(scene, self.cfg.world, estable)
        self.grid = grid
        self.frame_idx += 1

        if not scene.car.visible:
            self.path, self.path_px = [], []

        self._planificar(scene, grid)
        clearance = holgura_frente(
            scene.obstacle_mask,
            scene.car,
            max(1.0, self.cfg.control.stop_distance_px * 8.0),
        )
        drive = self.pursuit.update(scene.car, self.path_px, clearance, scene.obstacle_mask)

        now = self.clock
        if now < self._escape_until:
            spin = int(self.cfg.control.min_pwm * 1.9)
            slow = int(spin * 0.6)
            left, right = (-spin, -slow) if self._escape_dir > 0 else (-slow, -spin)
            drive = Drive(left=left, right=right, state=ESCAPE, clearance=clearance)
            self._blocked_since = None
        elif drive.state == BLOCKED:
            if self._blocked_since is None:
                self._blocked_since = now
            elif now - self._blocked_since > 0.45:
                self._start_escape(now)
        else:
            self._blocked_since = None

        self._chequear_meta(scene)
        if self.at_goal:
            drive = Drive(state=AT_GOAL, clearance=clearance)

        if drive.state not in (STOP, AT_GOAL) and self.speed_scale < 1.0:
            drive = drive.scaled(self.speed_scale)

        self._track_and_stuck(scene, drive, now)
        self._ai_tick(scene, grid, drive, now)

        if self.paused or self.estop:
            drive = Drive(state=PAUSED if self.paused else "FRENO", clearance=clearance)
        self.drive = drive
        self.link.send(drive.left, drive.right)

    def _planificar(self, scene: Scene, grid: Grid) -> None:
        if not scene.car.visible or grid.car_cell is None or self.at_goal:
            return
        now = self.clock
        if now < self._plan_blocked_until:
            return
        target_cell = self.ai_goal_cell or grid.to_cell(*self.goal_px)
        cambio_meta = target_cell != self._planned_goal
        if not cambio_meta and self.path:
            if self._ruta_sigue_sirviendo(grid, scene.car):
                return
        if not cambio_meta and (now - self._planned_at) < self.cfg.planner.min_interval_s:
            return
        started = time.perf_counter()
        path = plan_smooth(grid, grid.car_cell, target_cell, self.cfg.planner.max_expansions)
        if not path and grid.raw_blocked is not None:
            relaxed = Grid(
                free=~grid.raw_blocked,
                blocked=grid.raw_blocked,
                cell_px=grid.cell_px,
                car_cell=grid.car_cell,
            )
            path = plan_smooth(relaxed, grid.car_cell, target_cell, self.cfg.planner.max_expansions)
            if path:
                self.events.append("ruta al ras")
        self.stats.planner_ms = (time.perf_counter() - started) * 1000.0
        self.stats.replans += 1
        self._planned_at = now
        self._planned_goal = target_cell
        if path:
            self.path = path
            self.path_px = [grid.to_px(cell) for cell in path]
        else:
            self.path, self.path_px = [], []
            self._plan_blocked_until = now + 0.5

    def _ruta_sigue_sirviendo(self, grid: Grid, pose) -> bool:
        if len(self.path) < 2:
            return False
        cercano = 0
        mejor = 1e18
        for indice, (px, py) in enumerate(self.path_px):
            distancia = (px - pose.x) ** 2 + (py - pose.y) ** 2
            if distancia < mejor:
                mejor, cercano = distancia, indice
        if math.sqrt(mejor) > grid.cell_px * 3.0:
            return False
        tramo = self.path[cercano : cercano + 8]
        return all(grid.walkable(celda) for celda in tramo)

    def _chequear_meta(self, scene: Scene) -> None:
        if not scene.car.visible or self.at_goal:
            return
        reached_px = max(10.0, self.cfg.world.cell_px * 1.6)
        if self.ai_goal_cell is not None:
            gx, gy = self.grid.to_px(self.ai_goal_cell)
            if math.hypot(scene.car.x - gx, scene.car.y - gy) < reached_px:
                self.ai_goal_cell = None
                self.path = []
                self.events.append("objetivo IA alcanzado")
        gx, gy = self.goal_px
        dist = math.hypot(scene.car.x - gx, scene.car.y - gy)
        self.stats.min_goal_dist = min(self.stats.min_goal_dist, dist)
        if dist < reached_px:
            self.at_goal = True
            self.stats.goal_reached = True
            self.events.append("META alcanzada")

    def _start_escape(self, now: float) -> None:
        self._escape_until = now + 0.7
        self._escape_dir *= -1
        self._blocked_since = None
        self.path, self.path_px = [], []
        self.events.append("salida de emergencia")

    def _track_and_stuck(self, scene: Scene, drive: Drive, now: float) -> None:
        if not scene.car.visible:
            self._stuck_ref = None
            return
        if self._stuck_ref is None:
            self._stuck_ref = (now, scene.car.x, scene.car.y)
            return
        started, x0, y0 = self._stuck_ref
        age = now - started
        if age < 2.5:
            return
        moved = math.hypot(scene.car.x - x0, scene.car.y - y0)
        watching = drive.state in (FOLLOW, BLOCKED, ESCAPE, NO_PATH)
        if moved > 10.0 or not watching:
            self._stuck_ref = (now, scene.car.x, scene.car.y)
            return

        self.stats.stuck_events += 1
        self.events.append(f"trabado ({moved:.0f} px en {age:.1f}s)")
        self._stuck_ref = (now, scene.car.x, scene.car.y)
        if self.cfg.llm.on_stuck:
            self._pedir_ia(scene, drive, now, force=True)
        self._start_escape(now)

    def _ai_context(self, scene: Scene, grid: Grid, drive: Drive) -> dict:
        goal_cell = grid.to_cell(*self.goal_px)
        ascii_map = render_ascii(
            grid, grid.car_cell, goal_cell, self.path, self.cfg.llm.ascii_width
        )
        return {
            "ascii_map": ascii_map,
            "car_cell": grid.car_cell,
            "goal_cell": goal_cell,
            "target_cell": self.ai_goal_cell or goal_cell,
            "state": drive.state,
            "speed": abs(drive.left) / 255.0,
            "obstacles": scene.obstacle_count,
            "events": list(self.events)[-3:],
        }

    def _pedir_ia(self, scene: Scene, drive: Drive, now: float, force: bool = False) -> bool:
        if not self.ai.available() or self.grid is None or not scene.car.visible:
            return False
        if not force and not self.cfg.llm.on_stuck:
            return False
        queued = self.ai.ask(self._ai_context(scene, self.grid, drive), force=force)
        if queued:
            self.last_ai_at = now
            self.stats.ai_calls += 1
        return queued

    def _ai_tick(self, scene: Scene, grid: Grid, drive: Drive, now: float) -> None:
        decision = self.ai.poll()
        if decision is not None:
            self._aplicar_decision(decision, grid)
            if not decision.ok:
                self.stats.ai_errors += 1
        if not self.ai.available() or self.at_goal or self.paused:
            return
        period = self.cfg.llm.period_s
        if period > 0 and (now - self.last_ai_at) >= period:
            self._pedir_ia(scene, drive, now, force=True)

    def _aplicar_decision(self, decision: Decision, grid: Grid) -> None:
        self.ai_text = decision.summary()
        if not decision.ok:
            self.events.append(f"IA fallo: {decision.error}")
            return
        if decision.speed >= 0.2:
            self.speed_scale = decision.speed
        if decision.cell is None:
            return
        cell = nearest_walkable(grid, decision.cell)
        if cell is None:
            self.events.append("IA propuso celda bloqueada")
            return
        if cell != self.ai_goal_cell:
            self.ai_goal_cell = cell
            self.path, self.path_px = [], []
            self.events.append(f"IA apunta a {cell}")

    def loop(self) -> LoopStats:
        if not self.setup():
            self.shutdown()
            return self.stats
        limit = self.args.seconds or 0.0
        started = time.monotonic()
        prev = time.monotonic()
        last_log = started

        try:
            while True:
                now = time.monotonic()
                dt = self.args.fixed_dt if self.args.fixed_dt else max(1e-4, now - prev)
                prev = now
                if limit and self.clock > limit:
                    break

                if self.sim is not None:
                    self.sim.step(dt)
                frame = self.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.step(frame, dt)
                self.stats.frames += 1
                self._frame_times.append(now)
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    self._fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
                self.stats.fps = self._fps

                view = self.render()
                if self._writer is not None:
                    self._writer.write(view)
                if not self.args.headless:
                    cv2.imshow(self._window, view)
                    if not self.handle_key(cv2.waitKey(1) & 0xFF):
                        break
                elif now - last_log > 2.0:
                    last_log = now
                    print(
                        f"[iris] t={now - started:5.1f}s fps={self._fps:4.1f} "
                        f"estado={self.drive.state} meta={self.stats.goal_reached} "
                        f"choques={self.sim.collisions if self.sim else '-'}"
                    )
                if (
                    getattr(self.args, "trace", False)
                    and now - getattr(self, "_last_trace", 0.0) > 0.5
                ):
                    self._last_trace = now
                    car = self.scene.car if self.scene else None
                    angle = math.degrees(car.heading) if car else 0.0
                    pose = (
                        f"({car.x:5.1f},{car.y:5.1f}) r={angle:6.1f}"
                        if car and car.visible
                        else "sin carro"
                    )
                    gx, gy = self.goal_px
                    print(
                        f"  t={now - started:5.2f} {pose} meta=({gx:.0f},{gy:.0f}) "
                        f"ruta={len(self.path):2d} pwm={self.drive.left:4d}/"
                        f"{self.drive.right:4d} {self.drive.state}"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return self.stats

    def render(self):
        if self.scene is None:
            return None
        nota = ""
        if self.grid is not None and self.grid.blocked.mean() < 0.05:
            nota = "sin luz o pista fuera de cuadro"
        if self.link.status.connected is False and self.cfg.serial.enabled and self.sim is None:
            nota = nota or "arduino desconectado"
        data = HudData(
            fps=self.stats.fps,
            state=self.drive.state,
            route=len(self.path),
            ai_text="" if self.ai_text == "sin consultas todavia" else self.ai_text,
            paused=self.paused or self.estop,
            note=nota,
        )
        view = self.hud.dibujar(
            self.scene.proc,
            self.path_px,
            self.drive,
            self.goal_px,
            self.scene.car,
            self.scene.obstacles,
            data,
        )
        if self.args.record and self._writer is None:
            height, width = view.shape[:2]
            self._writer = cv2.VideoWriter(
                self.args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
            )
        return view

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.paused = not self.paused
            if self.paused:
                self.link.stop()
        elif key == ord("a"):
            now = self.clock
            if self.scene and self.grid:
                fired = self._pedir_ia(self.scene, self.drive, now, force=True)
                self.events.append("IA consultada" if fired else "IA ocupada o apagada")
        elif key == ord("r"):
            self.path, self.path_px = [], []
            self.ai_goal_cell = None
            self.at_goal = False
            self.stats.stuck_events = 0
            self.events.append("recalcular")
        elif key == ord("e"):
            self.estop = not self.estop
            self.events.append("freno" if self.estop else "freno liberado")
        return True

    def shutdown(self) -> None:
        try:
            self.link.stop()
        finally:
            self.link.close()
            self.ai.close()
            if self.camera is not None:
                self.camera.close()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if not self.args.headless:
                cv2.destroyAllWindows()

    def report(self) -> dict:
        data = {
            "frames": self.stats.frames,
            "fps": round(self.stats.fps, 2),
            "goal_reached": self.stats.goal_reached,
            "min_goal_dist_px": round(self.stats.min_goal_dist, 1)
            if self.stats.min_goal_dist < 1e8
            else None,
            "planner_ms": round(self.stats.planner_ms, 2),
            "replans": self.stats.replans,
            "ai_calls": self.stats.ai_calls,
            "ai_errors": self.stats.ai_errors,
            "stuck_events": self.stats.stuck_events,
            "final_state": self.drive.state,
            "seconds": round(self.clock, 1),
        }
        if self.sim is not None:
            data["collisions"] = self.sim.collisions
            data["car_pose"] = [round(v, 1) for v in self.sim.car]
        return data


def guardar_reporte(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

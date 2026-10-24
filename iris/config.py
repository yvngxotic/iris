from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"


@dataclass
class CameraCfg:
    source: Any = 0
    width: int = 640
    height: int = 480
    proc_width: int = 480
    fourcc: str = "MJPG"
    warmup_frames: int = 6
    lock_exposure: bool = False
    exposure: float = 120
    auto_wb: bool = True


@dataclass
class HsvRangeCfg:
    h_min: int = 85
    h_max: int = 140
    s_min: int = 70
    v_min: int = 45
    v_max: int = 255


@dataclass
class FloorCfg:
    v_min: int = 110
    s_max: int = 60
    auto: bool = False
    auto_factor: float = 0.72


@dataclass
class AutoCfg:
    white_percentile: float = 80.0
    floor_ratio: float = 0.55
    blue_margin: float = 0.035


@dataclass
class VisionCfg:
    car_hsv: HsvRangeCfg = field(default_factory=HsvRangeCfg)
    floor: FloorCfg = field(default_factory=FloorCfg)
    auto: AutoCfg = field(default_factory=AutoCfg)
    mode: str = "auto"
    min_car_area_px: int = 120
    min_obstacle_px: int = 200
    obstacle_dilate_px: int = 5


@dataclass
class WorldCfg:
    cell_px: int = 8
    obstacle_frac: float = 0.18
    floor_frac_min: float = 0.35
    safety_cells: int = 1


@dataclass
class PlannerCfg:
    min_interval_s: float = 0.7
    max_expansions: int = 20000
    goal: list[int] | None = None


@dataclass
class ControlCfg:
    max_pwm: int = 175
    min_pwm: int = 58
    lookahead_px: float = 24.0
    heading_gain: float = 1.35
    turn_slowdown: float = 0.6
    stop_distance_px: float = 24.0


@dataclass
class SerialCfg:
    enabled: bool = True
    port: str = "auto"
    baud: int = 115200
    invert_left: bool = False
    invert_right: bool = False
    watchdog_ms: int = 600
    rate_hz: float = 20.0


@dataclass
class LlmCfg:
    enabled: bool = True
    base_url: str = "https://api.commandcode.ai/provider/v1"
    api_key_env: str = "COMMANDCODE_API_KEY"
    models: dict[str, str] = field(
        default_factory=lambda: {
            "balanced": "deepseek/deepseek-v4.1-flash",
            "fast": "deepseek/deepseek-v4.1-flash-fast",
        }
    )
    use: str = "fast"
    thinking: bool = False
    temperature: float = 0.15
    max_tokens: int = 220
    timeout_s: float = 2.5
    min_interval_s: float = 6.0
    period_s: float = 12.0
    on_stuck: bool = True
    ascii_width: int = 40

    def model_id(self) -> str:
        return (
            self.models.get(self.use)
            or self.models.get("fast")
            or "deepseek/deepseek-v4.1-flash-fast"
        )

    def api_key(self) -> str:
        cargar_env()
        return os.environ.get(self.api_key_env, "").strip()


@dataclass
class SimCfg:
    arena_w: int = 640
    arena_h: int = 480
    car_len: int = 30
    car_wid: int = 18
    max_speed_px_s: float = 110.0
    max_turn_rad_s: float = 2.6


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    world: WorldCfg = field(default_factory=WorldCfg)
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    serial: SerialCfg = field(default_factory=SerialCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    sim: SimCfg = field(default_factory=SimCfg)


def _seccion(raw: dict, name: str, cls):
    values = dict(raw.get(name) or {})
    return cls(**values)


def cargar_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    vision_raw = dict(raw.get("vision") or {})
    vision = VisionCfg(
        car_hsv=HsvRangeCfg(**(vision_raw.pop("car_hsv", {}) or {})),
        floor=FloorCfg(**(vision_raw.pop("floor", {}) or {})),
        auto=AutoCfg(**(vision_raw.pop("auto", {}) or {})),
        **vision_raw,
    )

    return Config(
        camera=_seccion(raw, "camera", CameraCfg),
        vision=vision,
        world=_seccion(raw, "world", WorldCfg),
        planner=_seccion(raw, "planner", PlannerCfg),
        control=_seccion(raw, "control", ControlCfg),
        serial=_seccion(raw, "serial", SerialCfg),
        llm=_seccion(raw, "llm", LlmCfg),
        sim=_seccion(raw, "sim", SimCfg),
    )


def cargar_env(path: str | Path | None = None) -> None:
    env_path = Path(path) if path else ENV_PATH
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

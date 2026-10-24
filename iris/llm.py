from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass

import requests

from .config import LlmCfg

SYSTEM_PROMPT = (
    "Eres el planificador de alto nivel de un carro autonomo que se mueve sobre un piso negro visto por una camara.\n"
    "Recibes un mapa ASCII: '#' = obstaculo, '.' = piso libre, 'C' = el vehiculo, 'G' = meta, 'o' = ruta que ya calculo el piloto automatico.\n"
    "Decide una celda libre que acerque al vehiculo a la meta esquivando obstaculos. Prioriza no chocar sobre llegar rapido.\n"
    "Responde SOLO con un objeto JSON, sin texto extra ni bloques de codigo:\n"
    '{"cell": [columna, fila], "speed": 0.0, "why": "motivo corto"}\n'
    "Reglas: 'cell' debe ser una celda '.' dentro del mapa y cerca de la meta; usa null si la ruta actual ya sirve. "
    "'speed' va de 0.2 a 1.0 y baja en pasillos angostos. 'why' es una frase de maximo 12 palabras, en espanol sin acentos."
)

USER_TEMPLATE = (
    "Mapa (columnas 0-{max_col}, filas 0-{max_row}):\n{ascii_map}\n\n"
    "Vehiculo en {car} | meta en {goal} | objetivo vigente {target}\n"
    "Estado: {state} | velocidad mandada {speed:.2f} | obstaculos detectados {obstacles}\n"
    "Eventos recientes: {events}\n"
    "Elige la proxima celda objetivo."
)


@dataclass
class Decision:
    ok: bool = False
    cell: tuple[int, int] | None = None
    speed: float = 1.0
    why: str = ""
    latency_ms: int = 0
    error: str = ""
    model: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"IA sin respuesta ({self.error})" if self.error else "IA sin respuesta"
        where = f"celda {self.cell[0]},{self.cell[1]}" if self.cell else "mantener ruta"
        return f"{where} x{self.speed:.2f} | {self.why}"


def build_user_prompt(ctx: dict) -> str:
    ascii_map = ctx.get("ascii_map", "")
    rows = ascii_map.splitlines() or [""]
    return USER_TEMPLATE.format(
        max_col=len(rows[0]) - 1,
        max_row=len(rows) - 1,
        ascii_map=ascii_map,
        car=ctx.get("car_cell"),
        goal=ctx.get("goal_cell"),
        target=ctx.get("target_cell"),
        state=ctx.get("state", "?"),
        speed=float(ctx.get("speed", 1.0)),
        obstacles=int(ctx.get("obstacles", 0)),
        events="; ".join(ctx.get("events") or ["ninguno"]),
    )


def parse_decision(content: str, model: str, latency_ms: int) -> Decision:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        return Decision(ok=False, error="respuesta sin JSON", latency_ms=latency_ms, model=model)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Decision(ok=False, error="JSON invalido", latency_ms=latency_ms, model=model)
    if not isinstance(data, dict):
        return Decision(ok=False, error="JSON no es objeto", latency_ms=latency_ms, model=model)

    cell = data.get("cell")
    parsed_cell = None
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        try:
            parsed_cell = (max(0, int(cell[0])), max(0, int(cell[1])))
        except (TypeError, ValueError):
            parsed_cell = None

    try:
        speed = float(data.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0

    why = str(data.get("why", "")).strip()[:60]
    return Decision(
        ok=True,
        cell=parsed_cell,
        speed=max(0.2, min(1.0, speed)),
        why=why,
        latency_ms=latency_ms,
        model=model,
    )


class Strategist:
    def __init__(self, cfg: LlmCfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.api_key = cfg.api_key() if self.enabled else ""
        self.model = cfg.model_id()
        self.last_decision: Decision | None = None
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._results: queue.Queue[Decision] = queue.Queue()
        self._in_flight = False
        self._last_ask = 0.0
        self._thread: threading.Thread | None = None
        self._running = False
        self._drop_extra = False

    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def busy(self) -> bool:
        return self._in_flight

    def start(self) -> None:
        if not self.available() or self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="iris-llm", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False

    def ask(self, ctx: dict, force: bool = False) -> bool:
        if not self.available() or self._in_flight:
            return False
        now = time.monotonic()
        if not force and (now - self._last_ask) < self.cfg.min_interval_s:
            return False
        try:
            self._queue.put_nowait(dict(ctx))
        except queue.Full:
            return False
        self._in_flight = True
        self._last_ask = now
        return True

    def poll(self) -> Decision | None:
        try:
            decision = self._results.get_nowait()
        except queue.Empty:
            return None
        self.last_decision = decision
        return decision

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "iris-car/1.0",
        }

    def _payload(self, ctx: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ctx)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        if not self.cfg.thinking and not self._drop_extra:
            body["reasoning_effort"] = "none"
        return body

    def _post(self, body: dict) -> requests.Response:
        return requests.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=(1.0, self.cfg.timeout_s),
        )

    def pedir(self, ctx: dict) -> Decision:
        started = time.monotonic()
        body = self._payload(ctx)
        try:
            response = self._post(body)
        except requests.Timeout:
            return Decision(
                ok=False,
                error="timeout",
                model=self.model,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except requests.RequestException as exc:
            return Decision(
                ok=False,
                error=exc.__class__.__name__,
                model=self.model,
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        if response.status_code == 400 and "reasoning_effort" in body:
            self._drop_extra = True
            body.pop("reasoning_effort", None)
            try:
                response = self._post(body)
            except requests.RequestException as exc:
                return Decision(
                    ok=False,
                    error=exc.__class__.__name__,
                    model=self.model,
                    latency_ms=int((time.monotonic() - started) * 1000),
                )

        latency = int((time.monotonic() - started) * 1000)
        if response.status_code != 200:
            detail = ""
            try:
                detail = str(response.json().get("error", ""))[:70]
            except ValueError:
                detail = response.text[:70]
            return Decision(
                ok=False,
                error=f"HTTP {response.status_code} {detail}".strip(),
                model=self.model,
                latency_ms=latency,
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return Decision(ok=False, error="respuesta rara", model=self.model, latency_ms=latency)

        return parse_decision(content, self.model, latency)

    def _worker(self) -> None:
        while self._running:
            try:
                ctx = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            decision = self.pedir(ctx)
            self._in_flight = False
            try:
                self._results.put_nowait(decision)
            except queue.Full:
                pass

    def pedir_prueba(self) -> Decision:
        ctx = {
            "ascii_map": "...G\n.C#.\n....",
            "car_cell": (1, 1),
            "goal_cell": (3, 0),
            "target_cell": (3, 0),
            "state": "prueba",
            "speed": 0.5,
            "obstacles": 1,
            "events": ["prueba de conexion"],
        }
        return self.pedir(ctx)

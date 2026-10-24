import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from iris.app import AT_GOAL, App
from iris.config import cargar_config
from run import build_parser


class FakeApi(BaseHTTPRequestHandler):
    reply = '{"cell": [30, 37], "speed": 0.7, "why": "baja por el hueco"}'
    hits = 0

    def do_POST(self):
        FakeApi.hits += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        data = json.dumps({"choices": [{"message": {"content": FakeApi.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def run_app(extra: list[str]):
    args = build_parser().parse_args(["--sim", "--headless"] + extra)
    cfg = cargar_config()
    app = App(cfg, args)
    app.loop()
    return app


def test_la_simulacion_llega_a_la_meta_sin_chocar():
    app = run_app(["--seconds", "16", "--no-llm"])
    stats = app.report()
    assert stats["goal_reached"], stats
    assert stats["collisions"] == 0, stats
    assert stats["fps"] > 20, stats
    assert app.drive.state == AT_GOAL
    assert app.stats.replans <= 6, f"recalculo la ruta {app.stats.replans} veces"


def test_sin_llm_el_carro_igual_anda():
    app = run_app(["--seconds", "6", "--no-llm"])
    assert app.report()["frames"] > 100
    assert app.stats.ai_calls == 0


@pytest.fixture
def fake_api(monkeypatch):
    FakeApi.hits = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("COMMANDCODE_API_KEY", "clave-de-prueba")
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_la_ia_consulta_y_entrega_objetivo(fake_api):
    args = build_parser().parse_args(["--sim", "--headless", "--seconds", "5"])
    cfg = cargar_config()
    cfg.llm.base_url = fake_api
    cfg.llm.period_s = 1.0
    cfg.llm.min_interval_s = 0.1
    app = App(cfg, args)
    app.loop()
    assert FakeApi.hits >= 1
    assert app.stats.ai_calls >= 1
    assert app.ai.last_decision is not None and app.ai.last_decision.ok
    assert "hueco" in app.ai_text
    assert app.ai_goal_cell is not None


def test_cambiar_la_meta_obliga_a_replanificar():
    args = build_parser().parse_args(["--sim", "--headless", "--seconds", "8", "--no-llm"])
    app = App(cargar_config(), args)
    original = app.step
    estado = {"cambiado": False, "rutas": []}

    def step(frame, dt):
        original(frame, dt)
        if app.path:
            estado["rutas"].append(tuple(app.path))
        if not estado["cambiado"] and app.stats.replans == 1 and app.path:
            estado["cambiado"] = True
            app.goal_px = (app.goal_px[0], app.goal_px[1] - 60)
            app.path = []
            app.path_px = []

    app.step = step
    app.loop()
    distintas = len(set(estado["rutas"]))
    assert estado["cambiado"]
    assert distintas <= 4, f"{distintas} rutas distintas en la corrida"

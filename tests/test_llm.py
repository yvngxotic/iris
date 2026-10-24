import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from iris.config import LlmCfg
from iris.llm import Strategist, build_user_prompt, parse_decision


class FakeApi(BaseHTTPRequestHandler):
    answers: list[tuple[int, dict]] = []
    requests: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeApi.requests.append(
            {
                "body": body,
                "path": self.path,
                "auth": self.headers.get("Authorization", ""),
            }
        )
        status, payload = FakeApi.answers.pop(0) if FakeApi.answers else (500, {"error": "vacio"})
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def api(monkeypatch):
    FakeApi.answers = []
    FakeApi.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COMMANDCODE_API_KEY", "clave-de-prueba")
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_prompt_incluye_el_mapa():
    prompt = build_user_prompt(
        {
            "ascii_map": ".C#\n..G",
            "car_cell": (1, 0),
            "goal_cell": (2, 1),
            "target_cell": (2, 1),
            "state": "SIGUIENDO",
            "speed": 0.7,
            "obstacles": 1,
            "events": ["nada"],
        }
    )
    assert ".C#" in prompt and "..G" in prompt
    assert "SIGUIENDO" in prompt
    assert "columnas 0-2" in prompt


def test_lee_json_limpio():
    decision = parse_decision('{"cell": [4, 7], "speed": 0.5, "why": "pasillo libre"}', "m", 120)
    assert decision.ok
    assert decision.cell == (4, 7)
    assert decision.speed == 0.5
    assert decision.why == "pasillo libre"


def test_lee_json_entre_texto():
    decision = parse_decision(
        'claro:\n```json\n{"cell": [2, 3], "speed": 1.4}\n```\nlisto', "m", 90
    )
    assert decision.ok
    assert decision.cell == (2, 3)
    assert decision.speed == 1.0


def test_celda_nula_mantiene_ruta():
    decision = parse_decision('{"cell": null, "speed": 0.9, "why": "ruta ok"}', "m", 80)
    assert decision.ok and decision.cell is None


def test_respuesta_basura_no_rompe():
    assert not parse_decision("no tengo idea", "m", 50).ok
    assert parse_decision('{"cell": "x"}', "m", 50).cell is None


def test_peticion_completa_contra_servidor(api):
    FakeApi.answers = [(200, completion('{"cell": [3, 4], "speed": 0.6, "why": "derecha"}'))]
    cfg = LlmCfg(base_url=api, timeout_s=3.0)
    strategist = Strategist(cfg)
    assert strategist.available()
    decision = strategist.pedir(
        {
            "ascii_map": "...\n.C.",
            "car_cell": (1, 1),
            "goal_cell": (2, 0),
            "target_cell": (2, 0),
            "state": "SIGUIENDO",
            "speed": 0.5,
            "obstacles": 1,
            "events": [],
        }
    )
    assert decision.ok
    assert decision.cell == (3, 4)
    sent = FakeApi.requests[0]
    assert sent["path"].endswith("/chat/completions")
    assert sent["auth"] == "Bearer clave-de-prueba"
    assert sent["body"]["model"] == cfg.model_id()
    assert sent["body"]["reasoning_effort"] == "none"
    assert sent["body"]["stream"] is False
    assert sent["body"]["messages"][0]["role"] == "system"


def test_reintenta_sin_reasoning_si_el_servidor_lo_rechaza(api):
    FakeApi.answers = [
        (400, {"error": "reasoning_effort not supported"}),
        (200, completion('{"cell": [1, 1], "speed": 0.8, "why": "ok"}')),
    ]
    strategist = Strategist(LlmCfg(base_url=api, timeout_s=3.0))
    decision = strategist.pedir(
        {
            "ascii_map": ".",
            "car_cell": (0, 0),
            "goal_cell": (0, 0),
            "target_cell": (0, 0),
            "state": "x",
            "speed": 0.5,
            "obstacles": 0,
            "events": [],
        }
    )
    assert decision.ok
    assert len(FakeApi.requests) == 2
    assert "reasoning_effort" not in FakeApi.requests[1]["body"]


def test_error_http_se_reporta(api):
    FakeApi.answers = [(401, {"error": "clave mala"})]
    strategist = Strategist(LlmCfg(base_url=api, timeout_s=3.0))
    decision = strategist.pedir(
        {
            "ascii_map": ".",
            "car_cell": (0, 0),
            "goal_cell": (0, 0),
            "target_cell": (0, 0),
            "state": "x",
            "speed": 0.5,
            "obstacles": 0,
            "events": [],
        }
    )
    assert not decision.ok
    assert "401" in decision.error


def test_sin_clave_no_esta_disponible(monkeypatch):
    monkeypatch.delenv("COMMANDCODE_API_KEY", raising=False)
    monkeypatch.setattr(LlmCfg, "api_key", lambda self: "")
    assert not Strategist(LlmCfg()).available()


def test_min_intervalo_evita_spam(api):
    FakeApi.answers = [(200, completion('{"cell": [1, 0], "speed": 0.5, "why": "ok"}'))] * 4
    strategist = Strategist(LlmCfg(base_url=api, min_interval_s=60.0))
    ctx = {
        "ascii_map": ".",
        "car_cell": (0, 0),
        "goal_cell": (0, 0),
        "target_cell": (0, 0),
        "state": "x",
        "speed": 0.5,
        "obstacles": 0,
        "events": [],
    }
    assert strategist.ask(ctx, force=True)
    assert not strategist.ask(ctx, force=False)

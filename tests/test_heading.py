import math

import numpy as np

from iris.app import App
from iris.config import cargar_config
from iris.sim import _quads_overlap
from run import build_parser


def wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def test_colision_entre_rectangulos_girados():
    barra_horizontal = np.array([[0, 8], [30, 8], [30, 16], [0, 16]], dtype=float)
    barra_vertical = np.array([[12, -10], [20, -10], [20, 30], [12, 30]], dtype=float)
    lejos = np.array([[40, 0], [60, 0], [60, 20], [40, 20]], dtype=float)
    assert _quads_overlap(barra_horizontal, barra_vertical)
    assert not _quads_overlap(barra_horizontal, lejos)


def test_el_rumbo_no_se_voltea_entre_cuadros():
    args = build_parser().parse_args(
        ["--sim", "--headless", "--seconds", "10", "--fixed-dt", "0.0166", "--no-llm"]
    )
    app = App(cargar_config(), args)
    rumbo = []
    original = app.step

    def step(frame, dt):
        original(frame, dt)
        if app.scene and app.scene.car.visible:
            rumbo.append(app.scene.car.heading)

    app.step = step
    app.loop()

    assert len(rumbo) > 100
    saltos = [abs(wrap(b - a)) for a, b in zip(rumbo[10:], rumbo[11:], strict=False)]
    assert max(saltos) < math.radians(100), f"salto maximo {math.degrees(max(saltos)):.0f} grados"

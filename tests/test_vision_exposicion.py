import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision

BLANCO = 235
NEGRO = (22, 22, 22)
AZUL = (210, 120, 55)


def escena() -> np.ndarray:
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (120, 160), (200, 220), AZUL, -1)
    cv2.rectangle(frame, (380, 300), (500, 420), NEGRO, -1)
    return frame


def exponer(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def balance(frame: np.ndarray, ganancias: tuple[float, float, float]) -> np.ndarray:
    salida = frame.astype(np.float32).copy()
    for canal, ganancia in enumerate(ganancias):
        salida[:, :, canal] *= ganancia
    return np.clip(salida, 0, 255).astype(np.uint8)


def analizar(frame: np.ndarray):
    scene = Vision(VisionCfg(), 480).analizar(frame)
    return scene


def test_aguanta_subexposicion():
    for factor in (0.35, 0.5, 0.7):
        scene = analizar(exponer(escena(), factor))
        assert scene.car.visible, f"sin carro con factor {factor}"
        assert 80 < scene.car.x < 160, f"carro mal ubicado con factor {factor}"
        assert scene.obstacle_count >= 1, f"sin obstaculo con factor {factor}"


def test_aguanta_sobreexposicion():
    for factor in (1.3, 1.6, 2.0):
        frame = exponer(escena(), factor)
        scene = analizar(frame)
        assert scene.car.visible, f"sin carro con factor {factor}"
        assert scene.obstacle_count >= 1, f"sin obstaculo con factor {factor}"


def test_aguanta_balance_de_blancos_frio():
    for ganancias in ((1.15, 1.0, 0.85), (1.3, 1.0, 0.75)):
        scene = analizar(balance(escena(), ganancias))
        assert scene.car.visible, f"sin carro con balance {ganancias}"


def test_aguanta_balance_de_blancos_calido():
    for ganancias in ((0.85, 1.0, 1.15), (0.75, 1.0, 1.3)):
        scene = analizar(balance(escena(), ganancias))
        assert scene.car.visible, f"sin carro con balance {ganancias}"


def test_aguanta_todo_junto():
    combinaciones = [
        (0.45, (1.2, 1.0, 0.8)),
        (0.6, (0.8, 1.0, 1.2)),
        (1.4, (1.25, 1.0, 0.7)),
        (1.8, (0.7, 1.0, 1.25)),
    ]
    for factor, ganancias in combinaciones:
        frame = balance(exponer(escena(), factor), ganancias)
        scene = analizar(frame)
        assert scene.car.visible, f"sin carro con {factor} y {ganancias}"
        assert scene.obstacle_count >= 1, f"sin obstaculo con {factor} y {ganancias}"


def test_no_confunde_el_negro_con_el_carro():
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 330), NEGRO, -1)
    scene = analizar(frame)
    assert not scene.car.visible
    assert scene.obstacle_count >= 1


def test_no_confunde_el_blanco_con_el_carro():
    scene = analizar(np.full((480, 640, 3), BLANCO, dtype=np.uint8))
    assert not scene.car.visible


def test_el_piso_queda_libre_con_el_objeto_azul_encima():
    scene = analizar(escena())
    alto, ancho = scene.floor_mask.shape[:2]
    assert (scene.floor_mask > 0).mean() > 0.7
    assert scene.floor_mask[int(0.2 * alto), int(0.75 * ancho)] == 255


def test_la_sombra_suave_no_parte_el_piso():
    frame = escena()
    cv2.rectangle(frame, (300, 200), (560, 380), (150, 150, 150), -1)
    scene = analizar(frame)
    assert (scene.floor_mask > 0).mean() > 0.6

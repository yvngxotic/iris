import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision

BLANCO = 238
AZUL = (210, 120, 55)


def test_prefiere_forma_compacta_sobre_una_franja():
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (125, 320), AZUL, -1)
    cv2.rectangle(frame, (380, 220), (440, 270), AZUL, -1)
    scene = Vision(VisionCfg(), 480).analizar(frame)
    assert scene.car.visible
    assert scene.car.x > 250


def test_sin_forma_compacta_igual_elige_algo():
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (200, 60), (240, 380), AZUL, -1)
    scene = Vision(VisionCfg(), 480).analizar(frame)
    assert scene.car.visible


def test_no_elige_una_mancha_que_toca_el_borde():
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (120, 90), AZUL, -1)
    cv2.rectangle(frame, (300, 250), (350, 290), AZUL, -1)
    scene = Vision(VisionCfg(), 480).analizar(frame)
    assert scene.car.visible
    assert 230 < scene.car.x < 255


def test_si_todo_esta_cortado_por_el_borde_no_hay_vehiculo():
    frame = np.full((480, 640, 3), BLANCO, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (150, 100), AZUL, -1)
    cv2.rectangle(frame, (500, 380), (639, 479), AZUL, -1)
    scene = Vision(VisionCfg(), 480).analizar(frame)
    assert not scene.car.visible

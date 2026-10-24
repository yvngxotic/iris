import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision

WHITE = 235
AZUL = (200, 120, 60)


def frame_with_blobs(blobs):
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    for x0, y0, x1, y1 in blobs:
        cv2.rectangle(frame, (x0, y0), (x1, y1), AZUL, -1)
    return frame


def test_ignora_azul_pegado_al_borde():
    vision = Vision(VisionCfg(), 480)
    frame = frame_with_blobs([(0, 0, 90, 120), (300, 200, 350, 230)])
    scene = vision.analizar(frame)
    assert scene.car.visible
    assert scene.car.x > 100


def test_elige_el_mas_grande_entre_los_de_adentro():
    vision = Vision(VisionCfg(), 480)
    frame = frame_with_blobs([(200, 150, 240, 180), (300, 200, 360, 240)])
    scene = vision.analizar(frame)
    assert scene.car.visible
    assert 220 < scene.car.x < 280


def test_sigue_al_mismo_objeto_entre_cuadros():
    vision = Vision(VisionCfg(), 480)
    primero = frame_with_blobs([(120, 150, 170, 180)])
    scene = vision.analizar(primero)
    assert scene.car.visible
    arranque = scene.car.x

    segundo = frame_with_blobs([(140, 150, 190, 180), (480, 240, 560, 300)])
    scene = vision.analizar(segundo)
    assert scene.car.visible
    assert abs(scene.car.x - arranque) < 80


def test_sin_azul_no_inventa_vehiculo():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    assert not vision.analizar(frame).car.visible


def test_el_blanco_bajo_el_sol_no_es_azul():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), 253, dtype=np.uint8)
    assert not vision.analizar(frame).car.visible

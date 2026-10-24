import math

import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision, holgura_frente

BLUE = (255, 60, 30)
BLACK = (18, 18, 18)
WHITE = 235


def frame_with_scene(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), WHITE, dtype=np.uint8)
    cv2.rectangle(frame, (300, 150), (360, 270), BLACK, -1)
    cv2.rectangle(frame, (100, 100), (130, 118), BLUE, -1)
    return frame


def test_detecta_el_vehiculo_azul():
    vision = Vision(VisionCfg(), 480)
    scene = vision.analizar(frame_with_scene())
    assert scene.car.visible
    assert abs(scene.car.x - 115 * 0.75) < 6
    assert abs(scene.car.y - 109 * 0.75) < 6
    assert scene.car.half_len > scene.car.half_wid


def test_piso_blanco_es_la_mayoria():
    vision = Vision(VisionCfg(), 480)
    scene = vision.analizar(frame_with_scene())
    total = scene.floor_mask.size
    assert (scene.floor_mask > 0).sum() / total > 0.85


def test_obstaculo_negro_queda_marcado():
    vision = Vision(VisionCfg(), 480)
    scene = vision.analizar(frame_with_scene())
    px = int(330 * 0.75)
    py = int(210 * 0.75)
    assert scene.obstacle_mask[py, px] == 255
    assert scene.obstacle_count >= 1


def test_el_vehiculo_no_es_obstaculo():
    vision = Vision(VisionCfg(), 480)
    scene = vision.analizar(frame_with_scene())
    px = int(115 * 0.75)
    py = int(109 * 0.75)
    assert scene.obstacle_mask[py, px] == 0


def test_sin_carro_visible():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    scene = vision.analizar(frame)
    assert not scene.car.visible


def test_piso_oscuro_no_es_piso_en_modo_manual():
    cfg = VisionCfg(mode="manual")
    vision = Vision(cfg, 480)
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    scene = vision.analizar(frame)
    assert (scene.floor_mask > 0).sum() == 0


def test_rumbo_sigue_el_movimiento():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    for step in range(12):
        canvas = frame.copy()
        cv2.rectangle(canvas, (100 + step * 12, 100), (130 + step * 12, 118), BLUE, -1)
        scene = vision.analizar(canvas)
    assert scene.car.visible
    assert abs(math.degrees(scene.car.heading)) < 35


def test_rumbo_en_vertical():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    for step in range(12):
        canvas = frame.copy()
        cv2.rectangle(canvas, (200, 60 + step * 12), (218, 90 + step * 12), BLUE, -1)
        scene = vision.analizar(canvas)
    heading = math.degrees(scene.car.heading)
    assert 55 < abs(heading) < 125


def test_holgura_frontal_ve_el_muro():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    cv2.rectangle(frame, (300, 100), (320, 200), BLACK, -1)
    cv2.rectangle(frame, (200, 140), (230, 158), BLUE, -1)
    scene = vision.analizar(frame)
    clearance = holgura_frente(scene.obstacle_mask, scene.car, 200.0)
    assert 0 < clearance < 80


def test_sin_carro_no_hay_holgura():
    vision = Vision(VisionCfg(), 480)
    frame = np.full((480, 640, 3), WHITE, dtype=np.uint8)
    pose = vision.analizar(frame).car
    assert holgura_frente(np.zeros((360, 480), np.uint8), pose, 100.0) == 0.0

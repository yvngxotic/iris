import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision


def color_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    hsv = np.uint8([[[h, s, v]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def test_detecta_azul_palido_tipo_inhalador():
    cfg = VisionCfg()
    celeste = color_bgr(108, 94, 154)
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (260, 230), celeste, -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert scene.car.visible


def test_detecta_celeste_casi_gris():
    cfg = VisionCfg()
    apenas = color_bgr(108, 72, 150)
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (260, 230), apenas, -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert scene.car.visible


def test_no_confunde_el_blanco_con_el_carro():
    cfg = VisionCfg()
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (260, 230), (240, 238, 236), -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert not scene.car.visible


def test_billetera_negra_es_obstaculo():
    cfg = VisionCfg()
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (300, 120), (420, 260), (48, 48, 48), -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert scene.obstacle_count >= 1
    assert scene.obstacle_mask[int(190 * 0.75), int(360 * 0.75)] == 255


def test_piso_con_sombra_suave_sigue_siendo_piso():
    cfg = VisionCfg()
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (640, 480), (150, 150, 150), 120)
    scene = Vision(cfg, 480).analizar(frame)
    share = (scene.floor_mask > 0).mean()
    assert share > 0.5


def test_umbral_automatico_se_adapta_al_brillo():
    cfg = VisionCfg()
    cfg.floor.auto = True
    frame = np.full((480, 640, 3), 190, dtype=np.uint8)
    cv2.rectangle(frame, (300, 150), (360, 270), (18, 18, 18), -1)
    vision = Vision(cfg, 480)
    scene = vision.analizar(frame)
    assert vision.floor_v < 190
    assert scene.obstacle_count >= 1

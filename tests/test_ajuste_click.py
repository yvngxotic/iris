import cv2
import numpy as np

from iris.config import VisionCfg
from iris.vision import Vision, ajustar_rango


def parche(h: int, s: int, v: int) -> np.ndarray:
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
    mancha = np.full((15, 15, 3), bgr, dtype=np.uint8)
    return cv2.cvtColor(mancha, cv2.COLOR_BGR2HSV)


def test_el_click_ajusta_el_rango_al_objeto():
    ajuste = ajustar_rango(parche(108, 55, 130))
    assert ajuste["h_min"] <= 108 <= ajuste["h_max"]
    assert ajuste["s_min"] <= 55
    assert ajuste["v_min"] <= 130 <= ajuste["v_max"]


def test_el_click_en_un_objeto_palido_no_se_come_el_papel():
    ajuste = ajustar_rango(parche(108, 55, 130))
    papel = cv2.cvtColor(np.uint8([[[235, 235, 235]]]), cv2.COLOR_BGR2HSV)[0][0]
    h_papel, s_papel, v_papel = int(papel[0]), int(papel[1]), int(papel[2])
    entra = (
        ajuste["h_min"] <= h_papel <= ajuste["h_max"]
        and s_papel >= ajuste["s_min"]
        and ajuste["v_min"] <= v_papel <= ajuste["v_max"]
    )
    assert not entra


def test_el_click_en_azul_fuerte_deja_pasar_el_objeto():
    ajuste = ajustar_rango(parche(116, 210, 180))
    cfg = VisionCfg()
    cfg.car_hsv.__dict__.update(ajuste)
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    color = cv2.cvtColor(np.uint8([[[116, 210, 180]]]), cv2.COLOR_HSV2BGR)[0][0]
    cv2.rectangle(frame, (250, 200), (330, 240), tuple(int(c) for c in color), -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert scene.car.visible


def test_rango_ajustado_separa_objeto_de_papel_en_la_misma_escena():
    ajuste = ajustar_rango(parche(108, 60, 140))
    cfg = VisionCfg()
    cfg.car_hsv.__dict__.update(ajuste)
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    color = tuple(
        int(c) for c in cv2.cvtColor(np.uint8([[[108, 60, 140]]]), cv2.COLOR_HSV2BGR)[0][0]
    )
    cv2.rectangle(frame, (250, 200), (330, 240), color, -1)
    cv2.rectangle(frame, (420, 300), (520, 380), (40, 40, 40), -1)
    scene = Vision(cfg, 480).analizar(frame)
    assert scene.car.visible
    assert 205 < scene.car.x < 230
    assert scene.obstacle_count >= 1

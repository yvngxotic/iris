from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.camera import Camera
from iris.config import cargar_config
from iris.vision import ajustar_rango

WINDOW = "iris calibracion"
ULTIMO_CLICK = {"punto": None}


def nothing(_value: int) -> None:
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    if args.video:
        cfg.camera.source = args.video
    camera = Camera(cfg.camera)
    if not camera.open():
        print(f"no pude abrir la camara: {cfg.camera.source}")
        return 2

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("carro H min", WINDOW, cfg.vision.car_hsv.h_min, 179, nothing)
    cv2.createTrackbar("carro H max", WINDOW, cfg.vision.car_hsv.h_max, 179, nothing)
    cv2.createTrackbar("carro S min", WINDOW, cfg.vision.car_hsv.s_min, 255, nothing)
    cv2.createTrackbar("carro V min", WINDOW, cfg.vision.car_hsv.v_min, 255, nothing)
    cv2.createTrackbar("carro V max", WINDOW, cfg.vision.car_hsv.v_max, 255, nothing)
    cv2.createTrackbar("piso  V min", WINDOW, cfg.vision.floor.v_min, 255, nothing)
    cv2.createTrackbar("piso  S max", WINDOW, cfg.vision.floor.s_max, 255, nothing)
    cv2.createTrackbar("obstaculo px", WINDOW, cfg.vision.min_obstacle_px, 4000, nothing)

    def al_click(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            ULTIMO_CLICK["punto"] = (x, y)

    cv2.setMouseCallback(WINDOW, al_click)
    print("click sobre el vehiculo = ajusta el rango solo")
    print("m = guardar en config.yaml, q = salir")
    while True:
        frame = camera.read()
        if frame is None:
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue

        punto = ULTIMO_CLICK["punto"]
        if punto is not None:
            ULTIMO_CLICK["punto"] = None
            px, py = punto
            hsv_click = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            alto, ancho = hsv_click.shape[:2]
            if 0 <= px < ancho and 0 <= py < alto:
                x0, x1 = max(0, px - 7), min(ancho, px + 8)
                y0, y1 = max(0, py - 7), min(alto, py + 8)
                ajuste = ajustar_rango(hsv_click[y0:y1, x0:x1])
                cv2.setTrackbarPos("carro H min", WINDOW, ajuste["h_min"])
                cv2.setTrackbarPos("carro H max", WINDOW, ajuste["h_max"])
                cv2.setTrackbarPos("carro S min", WINDOW, ajuste["s_min"])
                cv2.setTrackbarPos("carro V min", WINDOW, ajuste["v_min"])
                cv2.setTrackbarPos("carro V max", WINDOW, ajuste["v_max"])
                print(f"rango ajustado al click: {ajuste}")

        car_h_min = cv2.getTrackbarPos("carro H min", WINDOW)
        car_h_max = cv2.getTrackbarPos("carro H max", WINDOW)
        car_s_min = cv2.getTrackbarPos("carro S min", WINDOW)
        car_v_min = cv2.getTrackbarPos("carro V min", WINDOW)
        car_v_max = cv2.getTrackbarPos("carro V max", WINDOW)
        floor_v = cv2.getTrackbarPos("piso  V min", WINDOW)
        floor_s = cv2.getTrackbarPos("piso  S max", WINDOW)
        min_px = cv2.getTrackbarPos("obstaculo px", WINDOW)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara_carro = cv2.inRange(
            hsv,
            np.array([car_h_min, car_s_min, car_v_min]),
            np.array([car_h_max, 255, 255]),
        )
        mascara_carro = cv2.morphologyEx(
            mascara_carro,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        h_ch, s_ch, v_ch = cv2.split(hsv)
        floor_mask = ((v_ch >= floor_v) & (s_ch <= floor_s)).astype(np.uint8) * 255
        obstacle = cv2.bitwise_not(cv2.bitwise_or(floor_mask, mascara_carro))

        contours, _ = cv2.findContours(mascara_carro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = max((cv2.contourArea(c) for c in contours), default=0.0)
        view = frame.copy()
        view[mascara_carro > 0] = (0, 255, 0)
        view[obstacle > 0] = view[obstacle > 0] // 2
        text = f"carro area {area:6.0f} px   {'OK' if area > 100 else 'NO LO VEO'}"
        cv2.putText(
            view,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if area > 100 else (0, 0, 255),
            2,
        )

        cv2.imshow(WINDOW, np.hstack([view, cv2.cvtColor(obstacle, cv2.COLOR_GRAY2BGR)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            path = (
                Path(args.config)
                if args.config
                else Path(__file__).resolve().parent.parent / "config.yaml"
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("vision", {})
            data["vision"]["car_hsv"] = {
                "h_min": car_h_min,
                "h_max": car_h_max,
                "s_min": car_s_min,
                "v_min": car_v_min,
                "v_max": car_v_max,
            }
            data["vision"]["floor"] = {"v_min": floor_v, "s_max": floor_s}
            data["vision"]["min_obstacle_px"] = min_px
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            print(f"guardado en {path}")

    camera.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

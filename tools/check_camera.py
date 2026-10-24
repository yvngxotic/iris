from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.camera import Camera
from iris.config import cargar_config
from iris.vision import Vision, _touches_border


def ok(cond: bool, good: str, bad: str) -> str:
    return f"[ OK ] {good}" if cond else f"[FALLA] {bad}"


def who_holds(device: str) -> list[str]:
    holders: list[str] = []
    target = os.path.realpath(device)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            handles = list((entry / "fd").iterdir())
        except OSError:
            continue
        for handle in handles:
            try:
                link = os.readlink(handle)
            except OSError:
                continue
            if link != target:
                continue
            try:
                name = (entry / "comm").read_text().strip()
            except OSError:
                name = "desconocido"
            holders.append(f"{name} (pid {entry.name})")
            break
    return holders


def explain_failure(source) -> None:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        print("  no hay ningun /dev/video*: la camara no esta conectada al PC")
        return
    for device in devices:
        holders = who_holds(str(device))
        if holders:
            print(f"  {device} esta tomada por {', '.join(holders)}")
        else:
            print(f"  {device} esta libre, pero no responde")
    if any(who_holds(str(d)) for d in devices):
        print("  cerra ese programa: una videollamada o una pestaña con la camara abierta")
        print("  alcanza para que la webcam quede ocupada en exclusiva")
    else:
        print("  proba desconectar y reconectar la camara, y revisa que el cable sea de datos")
    print(f"  fuente configurada: {source}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--salida", default="docs/diagnostico_camara.png")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    if args.video:
        cfg.camera.source = args.video

    camera = Camera(cfg.camera)
    if not camera.open():
        print(f"[FALLA] no pude abrir la camara {cfg.camera.source}")
        explain_failure(cfg.camera.source)
        return 2

    stats = camera.probe()
    print(f"camara       {stats.width}x{stats.height}  {stats.fourcc}  {stats.fps:.1f} fps reales")

    frame = camera.read()
    camera.close()
    if frame is None:
        print("[FALLA] la camara abrio pero no entrega cuadros")
        return 2

    vision = Vision(cfg.vision, cfg.camera.proc_width)
    scene = vision.analizar(frame)
    hsv = cv2.cvtColor(scene.proc, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    bright = float(value.mean())
    floor_share = float((scene.floor_mask > 0).mean())
    border = np.concatenate(
        [
            scene.floor_mask[0, :],
            scene.floor_mask[-1, :],
            scene.floor_mask[:, 0],
            scene.floor_mask[:, -1],
        ]
    )
    border_share = float((border > 0).mean())

    if bright < 45:
        tono = "muy oscuro"
    elif bright < 170:
        tono = "correcto"
    else:
        tono = "sobreexpuesto"
    print(f"brillo       {bright:.0f} sobre 255   ({tono})")
    modo = "auto (normalizado por cuadro)" if cfg.vision.mode == "auto" else "manual (HSV fijo)"
    print(f"modo         {modo}")
    print(f"referencia   blanco {vision.white_ref:.0f} de brillo, umbral de piso {vision.floor_v}")
    print(f"piso blanco  {floor_share * 100:.0f}% del cuadro")
    print(f"obstaculos   {scene.obstacle_count} manchas")
    if scene.car.visible and scene.car.contour is not None:
        area_carro = cv2.contourArea(scene.car.contour)
        donde = (
            f"({scene.car.x:.0f}, {scene.car.y:.0f}), "
            f"radio {scene.car.radius:.0f} px, area {area_carro:.0f} px"
        )
    else:
        area_carro = 0.0
        donde = "no aparece"
    print(f"carro azul   {donde}")
    mask = vision.mascara_carro(scene.proc)
    candidatos = sorted(vision.candidatos_carro(mask), key=cv2.contourArea, reverse=True)
    if candidatos:
        print("candidatos azules (el que se toma como vehiculo):")
        alto, ancho = mask.shape[:2]
        for c in candidatos[:5]:
            x, y, w, h = cv2.boundingRect(c)
            borde = ""
            if x <= 2 or y <= 2 or x + w >= ancho - 2 or y + h >= alto - 2:
                borde = "  (toca el borde, se descarta)"
            print(f"  area {cv2.contourArea(c):6.0f}  caja {w}x{h} en ({x},{y}){borde}")

    print()
    print(
        ok(
            bright >= 45,
            "hay suficiente luz para separar negro de obstaculos",
            "muy oscuro: el piso blanco tiene que estar iluminado para separarse del negro",
        )
    )
    print(
        ok(
            floor_share > 0.02,
            "se ve el piso blanco",
            "no hay blanco en el cuadro: apunta a la pista o prende la luz",
        )
    )
    print(
        ok(
            0.08 < floor_share < 0.92,
            "la pista ocupa una parte razonable del cuadro",
            "la pista esta de menos del 8% o de mas del 92%: acerca o aleja la camara",
        )
    )
    print(
        ok(
            border_share < 0.25,
            "la pista entra completa en el cuadro",
            "el borde del cuadro es piso: la camara no ve toda la pista, subila o alejala",
        )
    )
    print(
        ok(
            scene.obstacle_count >= 1,
            "se distinguen obstaculos sobre el piso",
            "no hay ninguna mancha que no sea piso: o no hay obstaculos o falta luz",
        )
    )
    if not scene.car.visible:
        print(
            "[FALLA] el vehiculo azul no aparece: pone el carro en la pista y revisa el rango azul"
        )
    elif scene.car.contour is not None and _touches_border(
        scene.car.contour, scene.proc.shape[1], scene.proc.shape[0]
    ):
        print(
            "[FALLA] el unico azul del cuadro toca el borde: no es el carro, "
            "o la camara no lo esta viendo entero"
        )
    elif area_carro < 600:
        print(
            f"[AVISO] el azul elegido es muy chico ({area_carro:.0f} px) para ser el carro: "
            "puede ser un reflejo. Poné el objeto azul sobre el papel"
        )
    else:
        print("[ OK ] el vehiculo azul esta ubicado dentro del cuadro")
    print(
        ok(
            stats.fps >= 12,
            f"{stats.fps:.0f} fps alcanzan para manejar",
            f"solo {stats.fps:.0f} fps: subi la luz, el sensor alarga la exposicion "
            "cuando esta oscuro",
        )
    )

    view = scene.proc.copy()
    view[scene.obstacle_mask > 0] = (0, 0, 220)
    if scene.car.visible and scene.car.contour is not None:
        cv2.drawContours(view, [scene.car.contour], -1, (0, 255, 0), 2)
    crudo = Path(args.salida).with_name("camara_actual.png")
    cv2.imwrite(str(crudo), scene.proc)
    mask = cv2.cvtColor(scene.obstacle_mask, cv2.COLOR_GRAY2BGR)
    pair = np.hstack([view, mask])
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pair)
    print(f"\nimagen anotada en {out}")
    print(f"cuadro crudo en {crudo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

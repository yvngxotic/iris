from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .control import Drive
from .vision import CarPose

ROSA = (255, 199, 214)
ROSA_OPACO = (214, 156, 176)
TINTA = (246, 241, 243)
APAGADO = (154, 144, 150)
ALERTA = (255, 146, 170)
FONDO = (11, 11, 12)
OBSTACULO = (108, 78, 88)
ROSA_LINEA = (127, 86, 224)
HALO = (32, 21, 42)

FUENTES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)
ALTO_FRANJA = 26
TAMANO = 13


def _fuente(tamano: int = TAMANO):
    for ruta in FUENTES:
        if Path(ruta).exists():
            return ImageFont.truetype(ruta, tamano)
    return ImageFont.load_default()


@dataclass
class HudData:
    fps: float = 0.0
    state: str = ""
    route: int = 0
    ai_text: str = ""
    paused: bool = False
    note: str = ""


class Hud:
    def __init__(self) -> None:
        self.fuente = _fuente()
        self.fuente_chica = _fuente(TAMANO - 2)

    def _franja(self, ancho: int, izquierda: list, derecha: list) -> np.ndarray:
        alto = ALTO_FRANJA
        imagen = Image.new("RGB", (ancho, alto), FONDO)
        dibujo = ImageDraw.Draw(imagen)
        x = 10
        for texto, color in izquierda:
            dibujo.text((x, 7), texto, font=self.fuente, fill=color)
            x += int(dibujo.textlength(texto, font=self.fuente)) + 14
        x = ancho - 10
        for texto, color in reversed(derecha):
            ancho_texto = int(dibujo.textlength(texto, font=self.fuente))
            x -= ancho_texto
            dibujo.text((x, 7), texto, font=self.fuente, fill=color)
            x -= 14
        return cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)

    def _pegar(self, vista: np.ndarray, franja: np.ndarray, y: int, opacidad: float) -> None:
        alto, ancho = franja.shape[:2]
        alto = min(alto, vista.shape[0] - y)
        region = vista[y : y + alto, 0:ancho]
        cv2.addWeighted(region, 1.0 - opacidad, franja[:alto], opacidad, 0, region)

    def dibujar(
        self,
        scene_proc: np.ndarray,
        path_px: list[tuple[float, float]],
        drive: Drive,
        goal_px: tuple[float, float] | None,
        car: CarPose,
        obstacles,
        data: HudData,
    ) -> np.ndarray:
        vista = scene_proc.copy()

        for contour in obstacles:
            cv2.drawContours(vista, [contour], -1, OBSTACULO, 1)

        if len(path_px) >= 2:
            puntos = np.array(path_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vista, [puntos], False, HALO, 4, cv2.LINE_AA)
            cv2.polylines(vista, [puntos], False, ROSA_LINEA, 2, cv2.LINE_AA)

        if goal_px:
            centro_meta = (int(goal_px[0]), int(goal_px[1]))
            cv2.circle(vista, centro_meta, 10, HALO, 4, cv2.LINE_AA)
            cv2.circle(vista, centro_meta, 10, ROSA_LINEA, 2, cv2.LINE_AA)
            cv2.drawMarker(vista, centro_meta, TINTA, cv2.MARKER_CROSS, 8, 1)

        if car.visible and car.contour is not None:
            cv2.drawContours(vista, [car.contour], -1, TINTA, 1)
            punta = (
                int(car.x + np.cos(car.heading) * (car.half_len + 9)),
                int(car.y + np.sin(car.heading) * (car.half_len + 9)),
            )
            cv2.arrowedLine(vista, (int(car.x), int(car.y)), punta, TINTA, 1, cv2.LINE_AA, 0, 0.4)

        alerta = drive.state in ("OBSTACULO", "SIN RUTA", "SIN VEHICULO")
        estado = "● " + data.state.lower()
        centro = [(data.ai_text[:34], ROSA_OPACO)] if data.ai_text else []
        if data.note:
            centro = [(data.note[:34], ALERTA)]

        arriba = self._franja(
            vista.shape[1],
            [("iris", ROSA)],
            [(f"{data.fps:.0f} fps", APAGADO)],
        )
        abajo = self._franja(
            vista.shape[1],
            [(estado, ALERTA if alerta else ROSA), *centro],
            [
                (f"{drive.left}/{drive.right}", APAGADO),
                (f"ruta {data.route}", APAGADO),
            ],
        )
        self._pegar(vista, arriba, 0, 0.82)
        self._pegar(vista, abajo, vista.shape[0] - ALTO_FRANJA, 0.82)

        if data.paused:
            alto, ancho = vista.shape[:2]
            cv2.putText(
                vista,
                "pausa",
                (ancho // 2 - 28, alto // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                FONDO,
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                vista,
                "pausa",
                (ancho // 2 - 28, alto // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                ROSA,
                1,
                cv2.LINE_AA,
            )
        return vista

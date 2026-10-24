from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import VisionCfg


@dataclass
class CarPose:
    visible: bool = False
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    radius: float = 14.0
    half_len: float = 15.0
    half_wid: float = 9.0
    confident: bool = False
    verified: bool = False
    contour: np.ndarray | None = None


@dataclass
class Scene:
    proc: np.ndarray
    floor_mask: np.ndarray
    obstacle_mask: np.ndarray
    car: CarPose
    obstacles: list[np.ndarray] = field(default_factory=list)

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacles)


def _touches_border(contour: np.ndarray, width: int, height: int, margin: int = 2) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def _axis_angle(contour: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 5:
        return 0.0
    _, eigenvectors = cv2.PCACompute(points, mean=None)
    axis = eigenvectors[0]
    return math.atan2(float(axis[1]), float(axis[0]))


class Vision:
    def __init__(self, cfg: VisionCfg, proc_width: int) -> None:
        self.cfg = cfg
        self.proc_width = proc_width
        self.car_lo = np.array([cfg.car_hsv.h_min, cfg.car_hsv.s_min, cfg.car_hsv.v_min])
        self.car_hi = np.array([cfg.car_hsv.h_max, 255, cfg.car_hsv.v_max])
        self.trail: deque[tuple[float, float]] = deque(maxlen=40)
        self._heading = 0.0
        self._motion: float | None = None
        self._verified = False
        self._flip_votes = 0
        self.floor_v = int(cfg.floor.v_min)
        self.white_ref = 255.0
        self._last_pos: tuple[float, float] | None = None
        self._ema: np.ndarray | None = None
        self.car_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.obs_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.gap_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def suavizar_obstaculos(self, mask: np.ndarray) -> np.ndarray:
        actual = mask.astype(np.float32)
        if self._ema is None or self._ema.shape != actual.shape:
            self._ema = actual
        else:
            self._ema = 0.6 * self._ema + 0.4 * actual
        return (self._ema >= 128).astype(np.uint8) * 255

    def mascara_carro(self, proc: np.ndarray) -> np.ndarray:
        azul, _ = self.clasificar(proc)
        return azul

    def clasificar(self, proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.mode == "manual":
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
            azul = cv2.inRange(hsv, self.car_lo, self.car_hi)
            piso = self._floor_manual(hsv)
            return self._limpiar(azul), piso

        azul_c = proc[:, :, 0].astype(np.int16)
        verde_c = proc[:, :, 1].astype(np.int16)
        rojo_c = proc[:, :, 2].astype(np.int16)
        suma = (azul_c + verde_c + rojo_c).astype(np.float32) + 1.0
        dominio = (azul_c - rojo_c) / suma
        brillo = np.maximum(np.maximum(azul_c, verde_c), rojo_c)

        muestra_brillo = brillo[::2, ::2]
        referencia = float(np.percentile(muestra_brillo, self.cfg.auto.white_percentile))
        referencia = max(8.0, referencia)
        self.white_ref = referencia
        self.floor_v = int(referencia * self.cfg.auto.floor_ratio)

        muestra_dominio = dominio[::2, ::2]
        claros = muestra_brillo >= referencia * 0.9
        if int(claros.sum()) < 30:
            claros = muestra_brillo >= float(np.percentile(muestra_brillo, 95))
        if int(claros.sum()) < 1:
            claros = np.ones_like(muestra_brillo, dtype=bool)
        papel = float(np.median(muestra_dominio[claros]))

        exceso = dominio - papel
        azul = (exceso > self.cfg.auto.blue_margin).astype(np.uint8) * 255
        piso = ((brillo >= self.floor_v) & (exceso <= self.cfg.auto.blue_margin)).astype(
            np.uint8
        ) * 255
        return self._limpiar(azul), piso

    def _floor_manual(self, hsv: np.ndarray) -> np.ndarray:
        _, s_ch, v_ch = cv2.split(hsv)
        floor_v = self._floor_threshold(v_ch)
        return ((v_ch >= floor_v) & (s_ch <= self.cfg.floor.s_max)).astype(np.uint8) * 255

    def _limpiar(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.car_open)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.car_open)

    def candidatos_carro(self, mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) >= self.cfg.min_car_area_px]

    @staticmethod
    def _looks_like_vehicle(
        contour: np.ndarray, max_aspect: float = 3.0, min_fill: float = 0.4
    ) -> bool:
        _, _, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return False
        if max(w, h) / float(min(w, h)) > max_aspect:
            return False
        return cv2.contourArea(contour) / float(w * h) >= min_fill

    def _choose_car(self, mask: np.ndarray) -> np.ndarray | None:
        blobs = self.candidatos_carro(mask)
        if not blobs:
            return None

        height, width = mask.shape[:2]
        compactos = [c for c in blobs if self._looks_like_vehicle(c)]
        if compactos:
            blobs = compactos
        inner = [c for c in blobs if not _touches_border(c, width, height)]
        if not inner:
            return None

        if self._last_pos is not None:
            max_jump = 0.25 * width
            near = min(inner, key=lambda c: self._distance(c, self._last_pos))
            if self._distance(near, self._last_pos) <= max_jump:
                return near
        return max(inner, key=cv2.contourArea)

    @staticmethod
    def _distance(contour: np.ndarray, point: tuple[float, float]) -> float:
        x, y, w, h = cv2.boundingRect(contour)
        return math.hypot(x + w / 2.0 - point[0], y + h / 2.0 - point[1])

    def _car(self, mask: np.ndarray) -> tuple[CarPose, np.ndarray]:
        contour = self._choose_car(mask)
        pose = CarPose()
        if contour is None:
            return pose, np.zeros_like(mask)

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return pose, mask
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        half_len = 0.5 * max(w, h, 1.0)
        half_wid = 0.5 * min(w, h, 1.0)
        radius = math.hypot(half_len, half_wid)

        pose.visible = True
        pose.x = float(cx)
        pose.y = float(cy)
        pose.radius = float(radius)
        pose.half_len = float(half_len)
        pose.half_wid = float(half_wid)
        pose.contour = contour
        pose.heading = self._heading
        self._last_pos = (float(cx), float(cy))

        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        return pose, filled

    def _update_heading(self, pose: CarPose) -> None:
        if not pose.visible or pose.contour is None:
            return
        self.trail.append((pose.x, pose.y))

        motion = None
        newest = self.trail[-1]
        for old in self.trail:
            dx = newest[0] - old[0]
            dy = newest[1] - old[1]
            if math.hypot(dx, dy) >= 9.0:
                motion = math.atan2(dy, dx)
                break

        if motion is not None:
            self._verified = True
            delta = abs(_wrap(motion - self._heading))
            if delta < math.radians(40):
                self._heading = _blend_angle(self._heading, motion, 0.5)
                self._flip_votes = 0
            elif delta > math.radians(120):
                self._flip_votes += 1
                if self._flip_votes >= 3:
                    self._heading = motion
                    self._flip_votes = 0
            else:
                self._flip_votes = 0
            self.trail.clear()
            self.trail.append(newest)

        axis = _axis_angle(pose.contour)
        flipped = _wrap(axis + math.pi)
        use_flipped = abs(_wrap(flipped - self._heading)) < abs(_wrap(axis - self._heading))
        self._heading = flipped if use_flipped else axis
        self._motion = motion if motion is not None else self._motion

        pose.heading = self._heading
        pose.confident = True
        pose.verified = self._verified

    def _floor_threshold(self, value_channel: np.ndarray) -> int:
        if not self.cfg.floor.auto:
            return int(self.cfg.floor.v_min)
        sample = value_channel[::4, ::4]
        paper = float(np.percentile(sample, 75))
        self.floor_v = int(np.clip(paper * self.cfg.floor.auto_factor, 55, 210))
        return self.floor_v

    def _obstacles(
        self, floor: np.ndarray, mascara_carro: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        obstacle = cv2.bitwise_not(cv2.bitwise_or(floor, mascara_carro))
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, self.obs_open)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle, connectivity=8)
        kept = np.zeros_like(obstacle)
        blobs: list[np.ndarray] = []
        for idx in range(1, count):
            if stats[idx, cv2.CC_STAT_AREA] < self.cfg.min_obstacle_px:
                continue
            blob = (labels == idx).astype(np.uint8) * 255
            kept = cv2.bitwise_or(kept, blob)
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                blobs.append(contours[0])
        if self.cfg.obstacle_dilate_px > 0:
            k = 2 * int(self.cfg.obstacle_dilate_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            kept = cv2.dilate(kept, kernel, iterations=1)
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, self.gap_close)
        return kept, blobs

    def analizar(self, frame: np.ndarray) -> Scene:
        scale = self.proc_width / float(frame.shape[1])
        proc = cv2.resize(
            frame,
            (self.proc_width, max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        mascara_carro, floor = self.clasificar(proc)
        car, mascara_carro = self._car(mascara_carro)
        floor = cv2.bitwise_and(floor, cv2.bitwise_not(mascara_carro))
        floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, self.obs_open)

        obstacle, blobs = self._obstacles(floor, mascara_carro)
        if car.visible:
            self._update_heading(car)

        return Scene(
            proc=proc,
            floor_mask=floor,
            obstacle_mask=obstacle,
            car=car,
            obstacles=blobs,
        )


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _blend_angle(current: float, target: float, weight: float) -> float:
    return _wrap(current + _wrap(target - current) * weight)


def holgura_frente(obstacle_mask: np.ndarray, pose: CarPose, max_px: float) -> float:
    if not pose.visible:
        return 0.0
    h, w = obstacle_mask.shape[:2]
    cos_h = math.cos(pose.heading)
    sin_h = math.sin(pose.heading)
    best = float(max_px)
    for offset in (-pose.half_wid * 0.75, 0.0, pose.half_wid * 0.75):
        hit = float(max_px)
        for advance in range(0, int(max_px) + 1, 3):
            reach = pose.half_len + advance
            px = int(round(pose.x + cos_h * reach - sin_h * offset))
            py = int(round(pose.y + sin_h * reach + cos_h * offset))
            if px < 0 or py < 0 or px >= w or py >= h:
                hit = float(advance)
                break
            if obstacle_mask[py, px] > 0:
                hit = float(advance)
                break
        best = min(best, hit)
    return best


def ajustar_rango(
    hsv_patch: np.ndarray,
    hue_margin: int = 28,
    sat_margin: int = 35,
    val_margin: int = 60,
) -> dict[str, int]:
    pixels = hsv_patch.reshape(-1, 3).astype(np.int32)
    hues = pixels[:, 0]
    weights = np.bincount(hues, minlength=180).astype(np.float64)
    window = np.convolve(np.concatenate([weights[-15:], weights, weights[:15]]), np.ones(9), "same")
    centro = int(np.argmax(window[15:195]))
    sat = int(np.median(pixels[:, 1]))
    val = int(np.median(pixels[:, 2]))
    return {
        "h_min": int(max(0, centro - hue_margin)),
        "h_max": int(min(179, centro + hue_margin)),
        "s_min": int(max(30, sat - sat_margin)),
        "v_min": int(max(0, val - val_margin)),
        "v_max": int(min(255, val + val_margin)),
    }

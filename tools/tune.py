from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.app import App
from iris.config import cargar_config
from run import build_parser


def corrida(cfg, seconds: float, dt: float) -> dict:
    args = build_parser().parse_args(
        [
            "--sim",
            "--headless",
            "--seconds",
            str(seconds),
            "--fixed-dt",
            str(dt),
            "--no-llm",
        ]
    )
    app = App(cfg, args)
    app.loop()
    data = app.report()
    data["tiempo_meta"] = round(app.clock, 2) if data["goal_reached"] else None
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1 / 60)
    parser.add_argument("--repeticiones", type=int, default=2)
    args = parser.parse_args()

    margenes = [1, 2]
    miradas = [22.0, 30.0, 38.0]
    ganancias = [1.15, 1.45, 1.75]
    resultados = []

    for margen, mirada, ganancia in itertools.product(margenes, miradas, ganancias):
        cfg = cargar_config()
        cfg.world.safety_cells = margen
        cfg.control.lookahead_px = mirada
        cfg.control.heading_gain = ganancia
        corridas = [corrida(cfg, args.seconds, args.dt) for _ in range(args.repeticiones)]
        llegadas = sum(1 for c in corridas if c["goal_reached"])
        choques = sum(c["collisions"] for c in corridas)
        tiempos = [c["tiempo_meta"] for c in corridas if c["tiempo_meta"]]
        promedio = round(sum(tiempos) / len(tiempos), 1) if tiempos else None
        resultados.append((llegadas, -choques, margen, mirada, ganancia, promedio, choques))
        print(
            f"margen={margen} mirada={mirada:4.0f} ganancia={ganancia:4.2f} "
            f"llegadas={llegadas}/{args.repeticiones} "
            f"choques={choques} tiempo={promedio}",
            flush=True,
        )

    resultados.sort(reverse=True)
    print("\nmejores:")
    for row in resultados[:5]:
        print(
            f"  llegadas={row[0]} choques={row[6]} margen={row[2]} "
            f"mirada={row[3]:4.0f} ganancia={row[4]:4.2f} tiempo={row[5]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

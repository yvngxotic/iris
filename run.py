from __future__ import annotations

import argparse
import sys

from iris.app import App, guardar_reporte
from iris.config import cargar_config
from iris.llm import Strategist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRIS - carro autonomo con vision y DeepSeek")
    parser.add_argument("--config", default=None, help="ruta a config.yaml")
    parser.add_argument("--sim", action="store_true", help="mundo simulado, sin camara ni arduino")
    parser.add_argument("--headless", action="store_true", help="sin ventana (para pruebas)")
    parser.add_argument("--seconds", type=float, default=0.0, help="corta a los N segundos")
    parser.add_argument("--report", default=None, help="escribe metricas JSON al salir")
    parser.add_argument("--record", default=None, help="graba el HUD a un mp4")
    parser.add_argument("--video", default=None, help="usa un archivo o URL como camara")
    parser.add_argument("--no-serial", action="store_true", help="no abre el puerto serie")
    parser.add_argument("--no-llm", action="store_true", help="no consulta la IA")
    parser.add_argument("--probe-llm", action="store_true", help="prueba la clave de la IA y sale")
    parser.add_argument("--goal", default=None, help="meta fija en pixeles de procesamiento: x,y")
    parser.add_argument(
        "--trace", action="store_true", help="imprime telemetria cada medio segundo"
    )
    parser.add_argument(
        "--fixed-dt",
        type=float,
        default=0.0,
        help="paso de tiempo fijo en segundos, solo simulacion, para correr mas rapido",
    )
    return parser


def _resumen(data: dict) -> str:
    segundos = data.get("seconds", 0.0)
    choques = data.get("collisions")
    partes = []
    if data.get("goal_reached"):
        partes.append(f"meta en {segundos:.0f}s")
    else:
        distancia = data.get("min_goal_dist_px")
        partes.append("sin meta" + (f", quedo a {distancia:.0f} px" if distancia else ""))
    if choques is not None:
        partes.append(f"{choques} choques")
    partes.append(f"{data.get('replans', 0)} recalculos")
    return ", ".join(partes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = cargar_config(args.config)

    if args.video:
        cfg.camera.source = args.video
    if args.no_serial:
        cfg.serial.enabled = False
    if args.no_llm:
        cfg.llm.enabled = False
    if args.goal:
        try:
            x_text, y_text = args.goal.split(",")
            cfg.planner.goal = [int(float(x_text)), int(float(y_text))]
        except ValueError:
            print("meta invalida, usa --goal x,y")
            return 2

    if args.probe_llm:
        strategist = Strategist(cfg.llm)
        if not strategist.available():
            print(f"[iris] falta la clave: exporta {cfg.llm.api_key_env} o crea .env")
            return 2
        print(f"[iris] probando {strategist.model} en {cfg.llm.base_url}")
        decision = strategist.pedir_prueba()
        if decision.ok:
            print(f"[iris] ok en {decision.latency_ms} ms -> {decision.summary()}")
            return 0
        print(f"[iris] fallo: {decision.error} ({decision.latency_ms} ms)")
        return 1

    app = App(cfg, args)
    try:
        app.loop()
    finally:
        data = app.report()
        if args.report:
            guardar_reporte(args.report, data)
        print("[iris] " + _resumen(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# IRIS - punto de entrada unico.
# Uso: ./iris.sh <comando>

set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "no encuentro el entorno en .venv"
  echo "crealo asi:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

comando="${1:-dev}"
shift || true

case "$comando" in
  dev|start|run)
    echo "== revisando camara =="
    "$PY" tools/check_camera.py || true
    echo
    echo "== arrancando =="
    exec "$PY" run.py "$@"
    ;;

  sim)
    exec "$PY" run.py --sim "$@"
    ;;

  demo)
    mkdir -p docs
    exec "$PY" run.py --sim --headless --seconds "${1:-14}" --record docs/demo_sim.mp4
    ;;

  check|camara)
    exec "$PY" tools/check_camera.py "$@"
    ;;

  calib|calibrar)
    exec "$PY" tools/calibrate.py "$@"
    ;;

  test|tests)
    exec "$PY" -m pytest tests -q "$@"
    ;;

  tune|afinar)
    exec "$PY" tools/tune.py "$@"
    ;;

  key|ia)
    exec "$PY" run.py --probe-llm "$@"
    ;;

  headless)
    segundos="${1:-10}"
    shift || true
    exec "$PY" run.py --headless --seconds "$segundos" "$@"
    ;;

  lint)
    exec "$PY" -m ruff check iris tools run.py tests "$@"
    ;;

  ayuda|help|-h|--help)
    cat <<'TXT'
./iris.sh              revisa la camara y arranca el carro
./iris.sh sim          simulador, sin camara ni arduino
./iris.sh check        diagnostico de camara y pista
./iris.sh calib        calibrador de color, un click sobre el vehiculo
./iris.sh test         corre los tests
./iris.sh key          prueba la clave de la API de la IA
./iris.sh tune         barrido de parametros de control
./iris.sh demo N       graba N segundos de simulacion en docs/demo_sim.mp4
./iris.sh headless 10  corre 10 segundos sin ventana
./iris.sh lint         revisa el estilo del codigo

Todo lo que va despues del comando se pasa tal cual a run.py, por ejemplo:
./iris.sh sim --trace --seconds 20
./iris.sh dev --video demo.mp4 --no-serial
TXT
    ;;

  *)
    echo "no conozco el comando '$comando'"
    echo
    exec "$0" help
    ;;
esac

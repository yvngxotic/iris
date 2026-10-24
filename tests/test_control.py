from iris.config import ControlCfg
from iris.control import BLOCKED, NO_PATH, STOP, Drive, PurePursuit
from iris.vision import CarPose


def pose(x=100.0, y=100.0, heading=0.0, verified=True):
    return CarPose(
        visible=True,
        x=x,
        y=y,
        heading=heading,
        radius=14.0,
        half_len=15.0,
        half_wid=9.0,
        confident=True,
        verified=verified,
    )


def test_sin_carro_frena():
    drive = PurePursuit(ControlCfg()).update(CarPose(), [], 100.0)
    assert drive.state == STOP
    assert drive.left == 0 and drive.right == 0


def test_sin_ruta_frena():
    drive = PurePursuit(ControlCfg()).update(pose(), [], 100.0)
    assert drive.state == NO_PATH


def test_obstaculo_cerca_frena():
    drive = PurePursuit(ControlCfg()).update(pose(), [(100, 100), (300, 100)], 10.0)
    assert drive.state == BLOCKED
    assert drive.left == 0 and drive.right == 0


def test_ruta_recta_va_parejo():
    drive = PurePursuit(ControlCfg()).update(pose(), [(100, 100), (200, 100), (300, 100)], 200.0)
    assert drive.left > 0 and drive.right > 0
    assert abs(drive.left - drive.right) <= 2


def test_objetivo_arriba_acelera_la_rueda_izquierda():
    drive = PurePursuit(ControlCfg()).update(pose(), [(100, 100), (100, 40)], 200.0)
    assert drive.left > drive.right


def test_objetivo_abajo_acelera_la_rueda_derecha():
    drive = PurePursuit(ControlCfg()).update(pose(), [(100, 100), (100, 160)], 200.0)
    assert drive.right > drive.left


def test_pwm_no_pasa_el_maximo():
    drive = PurePursuit(ControlCfg(max_pwm=150)).update(pose(), [(100, 100), (100, -500)], 200.0)
    assert max(abs(drive.left), abs(drive.right)) <= 150


def test_arranque_a_ciegas_avanza_derecho():
    drive = PurePursuit(ControlCfg()).update(
        pose(heading=0.0, verified=False), [(100, 100), (10, 100)], 200.0
    )
    assert drive.left == drive.right > 0
    assert drive.state == "CALIBRANDO"


def test_margen_de_paso_se_respeta():
    cfg = ControlCfg(stop_distance_px=24.0)
    libre = PurePursuit(cfg).update(pose(), [(100, 100), (200, 100)], 24.1)
    cerca = PurePursuit(cfg).update(pose(), [(100, 100), (200, 100)], 24.0)
    assert libre.left > 0
    assert cerca.left == 0


def test_escala_de_velocidad():
    drive = Drive(left=100, right=120, state="X")
    scaled = drive.scaled(0.5)
    assert scaled.left == 50 and scaled.right == 60

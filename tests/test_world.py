from iris.config import Config
from iris.sim import SimWorld
from iris.vision import Vision
from iris.world import armar_grid, render_ascii


def make_grid(cfg: Config = None):
    cfg = cfg or Config()
    world = SimWorld(cfg.sim)
    scene = Vision(cfg.vision, cfg.camera.proc_width).analizar(world.render())
    return scene, armar_grid(scene, cfg.world)


def test_la_celda_del_carro_es_transitable():
    scene, grid = make_grid()
    assert scene.car.visible
    assert grid.car_cell is not None
    assert grid.walkable(grid.car_cell)


def test_los_obstaculos_quedan_bloqueados():
    scene, grid = make_grid()
    cell = grid.cell_px
    for x, y, w, h, _ in SimWorld(Config().sim).obstacles[4:]:
        col = int((x * 0.75 + w * 0.375) // cell)
        row = int((y * 0.75 + h * 0.375) // cell)
        if 0 <= row < grid.rows and 0 <= col < grid.cols:
            assert grid.blocked[row, col]


def test_el_centro_del_pasillo_es_libre():
    scene, grid = make_grid()
    assert grid.walkable(grid.to_cell(150, 180))
    assert grid.walkable(grid.to_cell(240, 300))
    assert not grid.walkable(grid.to_cell(240, 250))


def test_ascii_tiene_mapa_y_carro():
    scene, grid = make_grid()
    text = render_ascii(grid, grid.car_cell, (55, 22), None, 40)
    assert "C" in text
    lines = text.splitlines()
    assert all(len(line) == 40 for line in lines)
    assert len(lines) >= 8


def test_el_inflado_no_tapa_todo():
    scene, grid = make_grid()
    free = grid.free.mean()
    assert 0.15 < free < 0.85

import numpy as np

from iris.planner import (
    line_of_sight,
    nearest_walkable,
    plan,
    plan_smooth,
    resample,
    smooth,
)
from iris.world import Grid, render_ascii


def grid_from(rows: list[str], cell_px: float = 8.0) -> Grid:
    blocked = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    return Grid(free=~blocked, blocked=blocked, cell_px=cell_px, car_cell=None)


MAPA = [
    "..........",
    "..........",
    "....##....",
    "....##....",
    "..........",
    "..........",
]


def test_ruta_esquiva_el_muro():
    grid = grid_from(MAPA)
    path = plan(grid, (0, 2), (9, 2))
    assert path is not None
    assert path[0] == (0, 2) and path[-1] == (9, 2)
    assert all(grid.walkable(cell) for cell in path)


def test_sin_ruta_cuando_esta_tapiado():
    grid = grid_from(
        [
            "...#....",
            "...#....",
            "...#....",
            "...#....",
            "...#....",
            "...#....",
        ]
    )
    assert plan(grid, (0, 2), (7, 2)) is None


def test_meta_bloqueada_usa_la_celda_libre_mas_cercana():
    grid = grid_from(MAPA)
    path = plan(grid, (0, 2), (4, 2))
    assert path is not None
    assert grid.walkable(path[-1])


def test_suavizado_no_mete_la_ruta_en_el_muro():
    grid = grid_from(MAPA)
    rough = plan(grid, (0, 2), (9, 2))
    soft = smooth(grid, rough)
    assert len(soft) <= len(rough)
    for cell in soft:
        assert grid.walkable(cell)


def test_linea_de_vista_bloqueada():
    grid = grid_from(MAPA)
    assert not line_of_sight(grid, (4, 0), (4, 5))
    assert line_of_sight(grid, (0, 0), (0, 5))


def test_resample_interpola():
    path = [(0, 0), (5, 0)]
    dense = resample(path, step=1.0)
    assert dense[0] == (0, 0) and dense[-1] == (5, 0)
    assert len(dense) >= 5


def test_celda_bloqueada_busca_vecina():
    grid = grid_from(MAPA)
    assert nearest_walkable(grid, (4, 2)) is not None
    assert grid.walkable(nearest_walkable(grid, (4, 2)))


def test_plan_smooth_devuelve_celdas_libres():
    grid = grid_from(MAPA)
    path = plan_smooth(grid, (0, 0), (9, 5))
    assert path
    assert all(grid.walkable(cell) for cell in path)


def test_ascii_marca_carro_y_meta():
    grid = grid_from(MAPA)
    text = render_ascii(grid, (0, 0), (9, 5), None, width=10)
    assert "C" in text and "G" in text
    assert "#" in text

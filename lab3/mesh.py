from __future__ import annotations

import numpy as np


def vertex_id(i: int, j: int, k: int, wy: int, wz: int) -> int:
    return i * (wy + 1) * (wz + 1) + j * (wz + 1) + k


def build_box_tet_mesh(wx: int, wy: int, wz: int, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for i in range(wx + 1):
        for j in range(wy + 1):
            for k in range(wz + 1):
                points.append([i * cell_size, j * cell_size, k * cell_size])

    tets = []
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                v000 = vertex_id(i, j, k, wy, wz)
                v001 = vertex_id(i, j, k + 1, wy, wz)
                v010 = vertex_id(i, j + 1, k, wy, wz)
                v011 = vertex_id(i, j + 1, k + 1, wy, wz)
                v100 = vertex_id(i + 1, j, k, wy, wz)
                v101 = vertex_id(i + 1, j, k + 1, wy, wz)
                v110 = vertex_id(i + 1, j + 1, k, wy, wz)
                v111 = vertex_id(i + 1, j + 1, k + 1, wy, wz)

                tets.extend(
                    [
                        [v000, v001, v011, v111],
                        [v000, v010, v011, v111],
                        [v000, v001, v101, v111],
                        [v000, v100, v101, v111],
                        [v000, v010, v110, v111],
                        [v000, v100, v110, v111],
                    ]
                )

    return np.asarray(points, dtype=np.float32), np.asarray(tets, dtype=np.int32)


def extract_unique_edges(tets: np.ndarray) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    local_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for tet in tets:
        for a, b in local_edges:
            i, j = int(tet[a]), int(tet[b])
            edge_set.add((i, j) if i < j else (j, i))
    edges = np.array(sorted(edge_set), dtype=np.int32)
    return edges

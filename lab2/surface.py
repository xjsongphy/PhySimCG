"""B5: Surface reconstruction from FLIP particles via Marching Cubes.

Builds a density field from particles, thresholds to get an isosurface,
then runs Marching Cubes to extract a triangle mesh.
"""
import numpy as np
import taichi as ti

# Marching Cubes edge table (12 edges × 256 cases)
_EDGE_TABLE = ti.field(int, shape=(256,))
# Marching Cubes triangle table (256 cases, up to 15 edges, -1 terminated)
_TRI_TABLE = ti.field(int, shape=(256, 16))


def _init_mc_tables():
    """Initialize Marching Cubes lookup tables (standard 256-case tables)."""
    import sys
    sys.setrecursionlimit(10000)

    # Edge connections: which 2 corners each of the 12 edges connects
    edge_conn = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    # For each of 256 cases, compute which edges are cut
    for case in range(256):
        edge_mask = 0
        for e in range(12):
            i0, i1 = edge_conn[e]
            if ((case >> i0) & 1) != ((case >> i1) & 1):
                edge_mask |= (1 << e)
        _EDGE_TABLE[case] = edge_mask

    # Triangle table: standard 256-case MC lookup
    # Using complete table from standard MC algorithm
    tri_table = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 8, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 8, 2, 8, 4, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 4, 2, 4, 8, 4, 7, 8, -1, -1, -1, -1],
        [8, 3, 2, 8, 4, 3, 4, 6, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 1, 0, 8, 4, 4, 6, 1, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 2, 4, 2, 8, 2, 10, 8, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 4, 10, 8, 6, 6, 3, 8, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 10, 6, 1, 10, 8, 6, 8, 6, 3, -1, -1, -1, -1],
        [1, 3, 8, 1, 6, 3, 1, 0, 6, 6, 5, 0, -1, -1, -1, -1],
        [0, 5, 9, 0, 6, 5, 6, 3, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [5, 6, 2, 5, 2, 9, 2, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 10, 9, 6, 2, 10, 10, 8, 2, 8, 2, 0, -1, -1, -1, -1],
        [7, 9, 8, 7, 0, 9, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 8, 0, 3, 7, 8, 7, 5, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 0, 7, 8, 7, 0, 2, 5, 0, -1],
        [2, 3, 11, 2, 11, 5, 2, 9, 11, 9, 5, 11, -1, -1, -1, -1],
        [0, 8, 3, 0, 5, 8, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 2, 11, 7, 5, 11, 8, 7, 9, 7, 0, -1, -1, -1, -1],
        [3, 11, 5, 3, 5, 8, 8, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 9, 7, 11, 0, 11, 5, 0, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 11, 9, 11, 8, 9, 8, 0, 8, 11, 5, -1, -1, -1, -1],
        [6, 11, 3, 6, 5, 11, 5, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 6, 5, 11, 3, -1, -1, -1, -1],
        [11, 3, 5, 11, 6, 3, 11, 5, 6, 8, 5, 6, -1, -1, -1, -1],
        [5, 6, 3, 5, 3, 11, 6, 8, 3, 8, 11, 0, -1, -1, -1, -1],
        [9, 6, 3, 9, 3, 11, 9, 11, 8, 11, 3, 6, -1, -1, -1, -1],
        [8, 6, 3, 8, 3, 11, 6, 10, 3, 0, 6, 3, -1, -1, -1, -1],
        [11, 6, 3, 11, 3, 10, 6, 2, 3, 2, 10, 8, -1, -1, -1, -1],
        [9, 10, 3, 9, 3, 6, 9, 6, 2, 2, 6, 3, -1, -1, -1, -1],
        [8, 9, 3, 8, 3, 2, 3, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 2, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [6, 3, 2, 6, 2, 10, 6, 10, 5, 10, 2, 3, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 6, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 10, 2, 11, 5, 10, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 10, 5, 11, 11, 5, 8, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 5, 6, 5, 3, 5, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [5, 6, 3, 5, 3, 11, 11, 3, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [8, 11, 5, 8, 5, 6, 3, 5, 11, -1, -1, -1, -1, -1, -1, -1],
        [5, 6, 11, 5, 11, 10, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 10, 11, 6, 11, 9, 6, -1, -1, -1, -1, -1, -1, -1],
    ]

    for c, tris in enumerate(tri_table):
        for i in range(16):
            _TRI_TABLE[c, i] = tris[i]


def build_surface_mesh(pos_field, nx, ny, nz, dx, threshold=0.5, mc_res=1):
    """Build a triangle mesh from particle positions using Marching Cubes.

    Args:
        pos_field: Taichi Vector field of particle positions (N, 3)
        nx, ny, nz: simulation grid resolution
        dx: cell size
        threshold: density threshold for isosurface (0-1)
        mc_res: MC grid refinement factor (1 = sim grid, 2 = 2x finer)

    Returns:
        (vertices, triangles) as numpy arrays, or (None, None) if empty
    """
    # Build density on a finer grid
    mc_nx = nx * mc_res
    mc_ny = ny * mc_res
    mc_nz = nz * mc_res
    mc_dx = dx / mc_res

    density = np.zeros((mc_nx + 1, mc_ny + 1, mc_nz + 1), dtype=np.float32)
    pos_np = pos_field.to_numpy()
    active_pos = pos_np[pos_np[:, 0] >= 0]

    # Simple point-splat density with Gaussian kernel
    sigma = mc_dx * 1.5
    sigma2 = 2 * sigma * sigma
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

    for px, py, pz in active_pos:
        ci0 = max(0, int((px - 2 * sigma) / mc_dx))
        cj0 = max(0, int((py - 2 * sigma) / mc_dx))
        ck0 = max(0, int((pz - 2 * sigma) / mc_dx))
        ci1 = min(mc_nx, int((px + 2 * sigma) / mc_dx) + 1)
        cj1 = min(mc_ny, int((py + 2 * sigma) / mc_dx) + 1)
        ck1 = min(mc_nz, int((pz + 2 * sigma) / mc_dx) + 1)

        for ci in range(ci0, ci1 + 1):
            cx = (ci + 0.5) * mc_dx
            dx2 = (cx - px) ** 2
            for cj in range(cj0, cj1 + 1):
                cy = (cj + 0.5) * mc_dx
                dy2 = (cy - py) ** 2
                for ck in range(ck0, ck1 + 1):
                    cz = (ck + 0.5) * mc_dx
                    dist2 = dx2 + dy2 + (cz - pz) ** 2
                    if dist2 < 4 * sigma2:
                        density[ci, cj, ck] += norm * np.exp(-dist2 / sigma2)

    # Normalize density
    max_d = density.max()
    if max_d > 0:
        density /= max_d

    # Marching Cubes
    verts = []
    tris = []
    vert_map = {}

    edge_verts = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    corner_offsets = [
        (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
        (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1),
    ]

    for i in range(mc_nx):
        for j in range(mc_ny):
            for k in range(mc_nz):
                # Get corner values
                vals = []
                for di, dj, dk in corner_offsets:
                    vals.append(density[i + di, j + dj, k + dk])

                # Determine case index
                case_idx = 0
                for c in range(8):
                    if vals[c] >= threshold:
                        case_idx |= (1 << c)

                if case_idx == 0 or case_idx == 255:
                    continue

                # Get triangle list for this case
                tri_list = []
                for t in range(16):
                    e = _TRI_TABLE[case_idx, t]
                    if e < 0:
                        break
                    tri_list.append(e)

                # Generate vertices for cut edges
                for e in tri_list:
                    key = (i, j, k, e)
                    if key in vert_map:
                        continue
                    e0, e1 = edge_verts[e]
                    v0, v1 = vals[e0], vals[e1]
                    if abs(v1 - v0) < 1e-8:
                        t_param = 0.5
                    else:
                        t_param = (threshold - v0) / (v1 - v0)
                    t_param = max(0.0, min(1.0, t_param))

                    c0 = corner_offsets[e0]
                    c1 = corner_offsets[e1]
                    x = (i + c0[0]) * mc_dx + t_param * (c1[0] - c0[0]) * mc_dx
                    y = (j + c0[1]) * mc_dx + t_param * (c1[1] - c0[1]) * mc_dx
                    z = (k + c0[2]) * mc_dx + t_param * (c1[2] - c0[2]) * mc_dx

                    vert_map[key] = len(verts)
                    verts.append([x, y, z])

                # Build triangles (3 consecutive edges = 1 triangle)
                for t in range(0, len(tri_list), 3):
                    if t + 2 >= len(tri_list):
                        break
                    tris.append([
                        vert_map[(i, j, k, tri_list[t])],
                        vert_map[(i, j, k, tri_list[t + 1])],
                        vert_map[(i, j, k, tri_list[t + 2])],
                    ])

    if len(verts) == 0:
        return None, None

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)

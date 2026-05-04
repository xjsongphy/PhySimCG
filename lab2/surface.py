"""B5: Surface reconstruction from FLIP particles via Marching Cubes."""
import numpy as np
import taichi as ti
from skimage.measure import marching_cubes


@ti.kernel
def _scatter_density(pos: ti.template(), density: ti.template(),
                     mc_nx: int, mc_ny: int, mc_nz: int,
                     mc_dx: float, sigma: float):
    sigma2 = 2.0 * sigma * sigma
    r = ti.cast(ti.ceil(3.0 * sigma / mc_dx), int)
    for p in range(pos.shape[0]):
        if pos[p][0] < 0:
            continue
        px, py, pz = pos[p]
        ci = ti.cast(px / mc_dx, int)
        cj = ti.cast(py / mc_dx, int)
        ck = ti.cast(pz / mc_dx, int)
        for di in range(-r, r + 1):
            ni = ci + di
            if ni < 0 or ni > mc_nx:
                continue
            cx = (ni + 0.5) * mc_dx
            dx2 = (cx - px) * (cx - px)
            for dj in range(-r, r + 1):
                nj = cj + dj
                if nj < 0 or nj > mc_ny:
                    continue
                cy = (nj + 0.5) * mc_dx
                dy2 = (cy - py) * (cy - py)
                dxy2 = dx2 + dy2
                if dxy2 > 9.0 * sigma2:
                    continue
                for dk in range(-r, r + 1):
                    nk = ck + dk
                    if nk < 0 or nk > mc_nz:
                        continue
                    cz = (nk + 0.5) * mc_dx
                    dist2 = dxy2 + (cz - pz) * (cz - pz)
                    if dist2 < 9.0 * sigma2:
                        density[ni, nj, nk] += ti.exp(-dist2 / sigma2)


def build_surface_mesh(pos_field, nx, ny, nz, dx, threshold=0.25, mc_res=1,
                       density_ti=None):
    """Build a triangle mesh from particle positions using Marching Cubes.

    Args:
        pos_field: Taichi Vector field of particle positions (N, 3)
        nx, ny, nz: simulation grid resolution
        dx: cell size
        threshold: density threshold for isosurface (0-1)
        mc_res: MC grid refinement factor (1 = sim grid, 2 = 2x finer)
        density_ti: optional pre-allocated Taichi scalar field for GPU scatter.
                    If provided, density is computed on GPU (much faster).

    Returns:
        (vertices, triangles) as numpy arrays, or (None, None) if empty
    """
    mc_nx = nx * mc_res
    mc_ny = ny * mc_res
    mc_nz = nz * mc_res
    mc_dx = dx / mc_res
    sigma = mc_dx * 2.5

    if density_ti is not None:
        density_ti.fill(0.0)
        _scatter_density(pos_field, density_ti, mc_nx, mc_ny, mc_nz, mc_dx, sigma)
        density = density_ti.to_numpy().astype(np.float64)
    else:
        pos_np = pos_field.to_numpy().astype(np.float64)
        active = pos_np[pos_np[:, 0] >= 0]
        if len(active) == 0:
            return None, None

        density = np.zeros((mc_nx + 1, mc_ny + 1, mc_nz + 1), dtype=np.float64)
        sigma2 = 2.0 * sigma * sigma
        r = int(np.ceil(3.0 * sigma / mc_dx))

        for px, py, pz in active:
            ci = int(px / mc_dx)
            cj = int(py / mc_dx)
            ck = int(pz / mc_dx)
            ci0 = max(0, ci - r)
            ci1 = min(mc_nx, ci + r)
            cj0 = max(0, cj - r)
            cj1 = min(mc_ny, cj + r)
            ck0 = max(0, ck - r)
            ck1 = min(mc_nz, ck + r)

            kk = np.arange(ck0, ck1 + 1)
            cz = (kk + 0.5) * mc_dx
            dz2 = (cz - pz) ** 2

            for i in range(ci0, ci1 + 1):
                cx = (i + 0.5) * mc_dx
                dx2 = (cx - px) ** 2
                if dx2 > 9.0 * sigma2:
                    continue
                for j in range(cj0, cj1 + 1):
                    cy = (j + 0.5) * mc_dx
                    dy2 = (cy - py) ** 2
                    dist2 = dx2 + dy2 + dz2
                    mask = dist2 < 9.0 * sigma2
                    if mask.any():
                        density[i, j, kk[mask]] += np.exp(-dist2[mask] / sigma2)

    max_d = density.max()
    if max_d == 0:
        return None, None
    density /= max_d

    spacing = np.array([mc_dx, mc_dx, mc_dx], dtype=np.float64)
    try:
        verts, faces, _, _ = marching_cubes(density, level=threshold, spacing=spacing)
    except (RuntimeError, ValueError):
        return None, None

    if len(verts) == 0 or len(faces) == 0:
        return None, None

    return verts.astype(np.float32), faces.astype(np.int32)

import taichi as ti
import taichi.math as tm
from lab2.flip import FLIPSimulator


@ti.data_oriented
class APICSimulator(FLIPSimulator):
    """Affine Particle-In-Cell (APIC) fluid simulator.

    Extends FLIP with affine velocity tensors (C_p) on each particle,
    preserving angular momentum during P2G and G2P transfers.
    """

    def __init__(self, nx: int, ny: int, nz: int, num_particles: int):
        super().__init__(nx, ny, nz, num_particles)
        # 3x3 affine matrix per particle (velocity gradient)
        self.C = ti.Matrix.field(3, 3, dtype=float, shape=num_particles)

    @ti.kernel
    def transfer_p2g(self):
        """P2G: scatter particle velocities + affine term to staggered grid faces."""
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]
            vx, vy, vz = self.vel[p]
            C = self.C[p]

            # --- u-component ---
            gx = px / dx
            gy = py / dx - 0.5
            gz = pz / dx - 0.5
            i0 = int(ti.floor(gx))
            j0 = int(ti.floor(gy))
            k0 = int(ti.floor(gz))
            fx = gx - i0
            fy = gy - j0
            fz = gz - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni <= self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
                    # Face position
                    fx_pos = ni * dx
                    fy_pos = (nj + 0.5) * dx
                    fz_pos = (nk + 0.5) * dx
                    # Affine velocity contribution
                    dpx = fx_pos - px
                    dpy = fy_pos - py
                    dpz = fz_pos - pz
                    affine_vx = vx + C[0, 0] * dpx + C[0, 1] * dpy + C[0, 2] * dpz
                    self.grid_u[ni, nj, nk] += w * affine_vx
                    self.grid_u_weight[ni, nj, nk] += w

            # --- v-component ---
            gx = px / dx - 0.5
            gy = py / dx
            gz = pz / dx - 0.5
            i0 = int(ti.floor(gx))
            j0 = int(ti.floor(gy))
            k0 = int(ti.floor(gz))
            fx = gx - i0
            fy = gy - j0
            fz = gz - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni < self.nx and 0 <= nj <= self.ny and 0 <= nk < self.nz:
                    w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
                    fx_pos = (ni + 0.5) * dx
                    fy_pos = nj * dx
                    fz_pos = (nk + 0.5) * dx
                    dpx = fx_pos - px
                    dpy = fy_pos - py
                    dpz = fz_pos - pz
                    affine_vy = vy + C[1, 0] * dpx + C[1, 1] * dpy + C[1, 2] * dpz
                    self.grid_v[ni, nj, nk] += w * affine_vy
                    self.grid_v_weight[ni, nj, nk] += w

            # --- w-component ---
            gx = px / dx - 0.5
            gy = py / dx - 0.5
            gz = pz / dx
            i0 = int(ti.floor(gx))
            j0 = int(ti.floor(gy))
            k0 = int(ti.floor(gz))
            fx = gx - i0
            fy = gy - j0
            fz = gz - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk <= self.nz:
                    w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
                    fx_pos = (ni + 0.5) * dx
                    fy_pos = (nj + 0.5) * dx
                    fz_pos = nk * dx
                    dpx = fx_pos - px
                    dpy = fy_pos - py
                    dpz = fz_pos - pz
                    affine_wz = vz + C[2, 0] * dpx + C[2, 1] * dpy + C[2, 2] * dpz
                    self.grid_w[ni, nj, nk] += w * affine_wz
                    self.grid_w_weight[ni, nj, nk] += w

    @ti.kernel
    def transfer_g2p(self, flip_ratio: float):
        """G2P: gather grid velocities and affine matrices to particles."""
        dx = self.dx
        dx2 = dx * dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]

            # PIC velocity from grid interpolation
            new_vel = self._interp_vel(self.grid_u, self.grid_v, self.grid_w, px, py, pz)
            old_grid_vel = self._interp_vel(self.grid_u_old, self.grid_v_old, self.grid_w_old, px, py, pz)
            delta = new_vel - old_grid_vel

            pic_vel = new_vel
            flip_vel = self.vel[p] + delta
            self.vel[p] = (1.0 - flip_ratio) * pic_vel + flip_ratio * flip_vel

            # Compute affine matrix C from finite differences on MAC grid
            ci = int(px / dx)
            cj = int(py / dx)
            ck = int(pz / dx)
            ci = ti.max(0, ti.min(self.nx - 2, ci))
            cj = ti.max(0, ti.min(self.ny - 2, cj))
            ck = ti.max(0, ti.min(self.nz - 2, ck))

            # Velocity gradient via central differences at cell centers
            du_dx = (self.grid_u[ci + 1, cj, ck] - self.grid_u[ci, cj, ck]) / dx
            du_dy = (self.grid_u[ci, cj + 1, ck] - self.grid_u[ci, cj, ck]) / dx
            du_dz = (self.grid_u[ci, cj, ck + 1] - self.grid_u[ci, cj, ck]) / dx
            dv_dx = (self.grid_v[ci + 1, cj, ck] - self.grid_v[ci, cj, ck]) / dx
            dv_dy = (self.grid_v[ci, cj + 1, ck] - self.grid_v[ci, cj, ck]) / dx
            dv_dz = (self.grid_v[ci, cj, ck + 1] - self.grid_v[ci, cj, ck]) / dx
            dw_dx = (self.grid_w[ci + 1, cj, ck] - self.grid_w[ci, cj, ck]) / dx
            dw_dy = (self.grid_w[ci, cj + 1, ck] - self.grid_w[ci, cj, ck]) / dx
            dw_dz = (self.grid_w[ci, cj, ck + 1] - self.grid_w[ci, cj, ck]) / dx

            new_C = tm.mat3(
                du_dx, du_dy, du_dz,
                dv_dx, dv_dy, dv_dz,
                dw_dx, dw_dy, dw_dz,
            )
            # FLIP-style blend for C
            self.C[p] = self.C[p] + (1.0 - flip_ratio) * (new_C - self.C[p])

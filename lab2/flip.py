import taichi as ti
import taichi.math as tm
from lab2.core import FluidSimulator


@ti.data_oriented
class FLIPSimulator(FluidSimulator):
    """PIC/FLIP hybrid fluid simulator on 3D staggered grid."""

    def substep(
        self,
        dt: float,
        flip_ratio: float = 0.95,
        gravity: float = -9.8,
        num_pressure_iters: int = 40,
        num_particle_iters: int = 2,
        over_relaxation: float = 1.9,
        compensate_drift: bool = True,
        separate_particles: bool = False,
    ):
        self.integrate_particles(dt, gravity)
        self.handle_particle_collisions()
        if separate_particles:
            self.push_particles_apart(num_particle_iters)
        self.handle_particle_collisions()

        # P2G
        self.clear_grid()
        self.transfer_p2g()
        self.normalize_grid_vel()

        # Save old grid vel for FLIP delta
        self.save_grid_vel_old()

        # Pressure solve
        self.update_particle_density()
        self.solve_incompressibility(
            num_pressure_iters, dt, over_relaxation, compensate_drift
        )
        self.set_boundary_velocity()

        # G2P
        self.transfer_g2p(flip_ratio)

    @ti.kernel
    def transfer_p2g(self):
        """Transfer particle velocities to staggered grid (P2G) with linear interp."""
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]
            vx, vy, vz = self.vel[p]

            # u-component: u[i,j,k] at position (i*dx, (j+0.5)*dx, (k+0.5)*dx)
            # Particle maps to u-grid index: ix = px/dx, iy = py/dx - 0.5, iz = pz/dx - 0.5
            ix_u = px / dx
            iy_u = py / dx - 0.5
            iz_u = pz / dx - 0.5
            i0 = int(ti.floor(ix_u))
            j0 = int(ti.floor(iy_u))
            k0 = int(ti.floor(iz_u))
            fx = ix_u - i0
            fy = iy_u - j0
            fz = iz_u - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni <= self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    wx = fx if di == 1 else (1.0 - fx)
                    wy = fy if dj == 1 else (1.0 - fy)
                    wz = fz if dk == 1 else (1.0 - fz)
                    w = wx * wy * wz
                    self.grid_u[ni, nj, nk] += w * vx
                    self.grid_u_weight[ni, nj, nk] += w

            # v-component: v[i,j,k] at position ((i+0.5)*dx, j*dx, (k+0.5)*dx)
            ix_v = px / dx - 0.5
            iy_v = py / dx
            iz_v = pz / dx - 0.5
            i0 = int(ti.floor(ix_v))
            j0 = int(ti.floor(iy_v))
            k0 = int(ti.floor(iz_v))
            fx = ix_v - i0
            fy = iy_v - j0
            fz = iz_v - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni < self.nx and 0 <= nj <= self.ny and 0 <= nk < self.nz:
                    wx = fx if di == 1 else (1.0 - fx)
                    wy = fy if dj == 1 else (1.0 - fy)
                    wz = fz if dk == 1 else (1.0 - fz)
                    w = wx * wy * wz
                    self.grid_v[ni, nj, nk] += w * vy
                    self.grid_v_weight[ni, nj, nk] += w

            # w-component: w[i,j,k] at position ((i+0.5)*dx, (j+0.5)*dx, k*dx)
            ix_w = px / dx - 0.5
            iy_w = py / dx - 0.5
            iz_w = pz / dx
            i0 = int(ti.floor(ix_w))
            j0 = int(ti.floor(iy_w))
            k0 = int(ti.floor(iz_w))
            fx = ix_w - i0
            fy = iy_w - j0
            fz = iz_w - k0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ni = i0 + di
                nj = j0 + dj
                nk = k0 + dk
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk <= self.nz:
                    wx = fx if di == 1 else (1.0 - fx)
                    wy = fy if dj == 1 else (1.0 - fy)
                    wz = fz if dk == 1 else (1.0 - fz)
                    w = wx * wy * wz
                    self.grid_w[ni, nj, nk] += w * vz
                    self.grid_w_weight[ni, nj, nk] += w

    @ti.kernel
    def transfer_g2p(self, flip_ratio: float):
        """Transfer grid velocities back to particles (G2P) with PIC/FLIP blend."""
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]
            old_vel = self.vel[p]

            # Interpolate new grid velocity at particle position (PIC)
            new_vel = self._interp_vel(self.grid_u, self.grid_v, self.grid_w, px, py, pz)
            # Interpolate old grid velocity (FLIP delta)
            old_grid_vel = self._interp_vel(self.grid_u_old, self.grid_v_old, self.grid_w_old, px, py, pz)
            delta = new_vel - old_grid_vel

            # Blend: v = (1-ratio)*PIC + ratio*(old_vel + FLIP_delta)
            pic_vel = new_vel
            flip_vel = old_vel + delta

            self.vel[p] = (1.0 - flip_ratio) * pic_vel + flip_ratio * flip_vel

    @ti.func
    def _interp_vel(self, u_field, v_field, w_field, px, py, pz) -> tm.vec3:
        """Interpolate full velocity from staggered grid at position (px,py,pz)."""
        vx = self._interp_staggered(u_field, px, py, pz,
                                     self.nx + 1, self.ny, self.nz,
                                     0.0, -0.5, -0.5)
        vy = self._interp_staggered(v_field, px, py, pz,
                                     self.nx, self.ny + 1, self.nz,
                                     -0.5, 0.0, -0.5)
        vz = self._interp_staggered(w_field, px, py, pz,
                                     self.nx, self.ny, self.nz + 1,
                                     -0.5, -0.5, 0.0)
        return tm.vec3(vx, vy, vz)

    @ti.func
    def _interp_staggered(self, field, px, py, pz,
                          shape_x, shape_y, shape_z,
                          off_x, off_y, off_z) -> float:
        """Interpolate a staggered grid component at (px,py,pz).

        The field has shape (shape_x, shape_y, shape_z).
        Grid node [i,j,k] is at position ((i+off_x+0.5)*dx, (j+off_y+0.5)*dx, (k+off_z+0.5)*dx)
        simplified: the fractional index in each dimension is pos/dx - offset.
        """
        dx = self.dx
        ix = px / dx - off_x
        iy = py / dx - off_y
        iz = pz / dx - off_z

        i0 = int(ti.floor(ix))
        j0 = int(ti.floor(iy))
        k0 = int(ti.floor(iz))

        fx = ix - i0
        fy = iy - j0
        fz = iz - k0

        result = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ni = ti.max(0, ti.min(shape_x - 1, i0 + di))
            nj = ti.max(0, ti.min(shape_y - 1, j0 + dj))
            nk = ti.max(0, ti.min(shape_z - 1, k0 + dk))

            wx = fx if di == 1 else (1.0 - fx)
            wy = fy if dj == 1 else (1.0 - fy)
            wz = fz if dk == 1 else (1.0 - fz)
            result += field[ni, nj, nk] * wx * wy * wz

        return result

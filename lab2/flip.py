import taichi as ti
import taichi.math as tm
from lab2.core import FluidSimulator


@ti.data_oriented
class FLIPSimulator(FluidSimulator):
    """PIC/FLIP hybrid fluid simulator on 3D staggered grid.

    Staggered grid convention (see core.py docstring):
    - u[i,j,k] at (i*dx, (j+0.5)*dx, (k+0.5)*dx)
    - v[i,j,k] at ((i+0.5)*dx, j*dx, (k+0.5)*dx)
    - w[i,j,k] at ((i+0.5)*dx, (j+0.5)*dx, k*dx)
    """

    def substep(
        self,
        dt: float,
        flip_ratio: float = 0.95,
        gravity: float = -9.8,
        num_pressure_iters: int = 40,
        num_particle_iters: int = 1,
        over_relaxation: float = 1.9,
        compensate_drift: bool = True,
        separate_particles: bool = True,
        reflection: bool = False,
    ):
        if reflection:
            self._substep_reflection(dt, flip_ratio, gravity,
                                     num_pressure_iters, num_particle_iters,
                                     over_relaxation, compensate_drift, separate_particles)
            return
        self.integrate_and_collide(dt, gravity)
        if separate_particles:
            self.push_particles_apart(num_particle_iters)
            self.integrate_and_collide(0.0, 0.0)  # collide only (dt=0)

        self.clear_grid()
        self.transfer_p2g()
        self.normalize_grid_vel()

        self.save_grid_vel_old()

        self.relabel_and_density()
        self.solve_incompressibility(
            num_pressure_iters, dt, over_relaxation, compensate_drift
        )
        self.set_boundary_velocity()

        self.transfer_g2p(flip_ratio)

    def _substep_reflection(self, dt, flip_ratio, gravity, np_iter, npart_iter,
                            over_rel, comp_drift, sep_parts):
        half_dt = dt * 0.5

        # --- First half: standard FLIP to mid-step ---
        self.integrate_and_collide(half_dt, gravity)
        if sep_parts:
            self.push_particles_apart(npart_iter)
            self.integrate_and_collide(0.0, 0.0)

        self.clear_grid()
        self.transfer_p2g()
        self.normalize_grid_vel()

        # Save advected grid velocity before projection
        self.save_grid_vel_old()

        self.relabel_and_density()
        self.solve_incompressibility(np_iter, half_dt, over_rel, comp_drift)
        self.set_boundary_velocity()

        # Reflect: u_reflected = 2*u_projected - u_advected
        self._reflect_grid_vel()

        # G2P: transfer reflected velocity to particles
        self.transfer_g2p(flip_ratio)

        # --- Second half: advect with reflected velocity ---
        self.integrate_and_collide(half_dt, 0.0)
        if sep_parts:
            self.push_particles_apart(npart_iter)
            self.integrate_and_collide(0.0, 0.0)

        self.clear_grid()
        self.transfer_p2g()
        self.normalize_grid_vel()

        self.relabel_and_density()
        self.solve_incompressibility(np_iter, half_dt, over_rel, comp_drift)
        self.set_boundary_velocity()

        self.transfer_g2p(flip_ratio)

    @ti.kernel
    def _reflect_grid_vel(self):
        """R = 2*P - I: reflect grid velocity across projection manifold."""
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            self.grid_u[i, j, k] = 2.0 * self.grid_u[i, j, k] - self.grid_u_old[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] = 2.0 * self.grid_v[i, j, k] - self.grid_v_old[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w[i, j, k] = 2.0 * self.grid_w[i, j, k] - self.grid_w_old[i, j, k]

    @ti.kernel
    def transfer_p2g(self):
        """P2G: scatter particle velocities to staggered grid faces with linear weights."""
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]
            vx, vy, vz = self.vel[p]

            # --- u-component ---
            # u[i,j,k] lives at (i*dx, (j+0.5)*dx, (k+0.5)*dx)
            # Particle position in u-grid coords: (px/dx, py/dx - 0.5, pz/dx - 0.5)
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
                    self.grid_u[ni, nj, nk] += w * vx
                    self.grid_u_weight[ni, nj, nk] += w

            # --- v-component ---
            # v[i,j,k] lives at ((i+0.5)*dx, j*dx, (k+0.5)*dx)
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
                    self.grid_v[ni, nj, nk] += w * vy
                    self.grid_v_weight[ni, nj, nk] += w

            # --- w-component ---
            # w[i,j,k] lives at ((i+0.5)*dx, (j+0.5)*dx, k*dx)
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
                    self.grid_w[ni, nj, nk] += w * vz
                    self.grid_w_weight[ni, nj, nk] += w

    @ti.kernel
    def transfer_g2p(self, flip_ratio: float):
        """G2P: gather grid velocities to particles with PIC/FLIP blend."""
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            px, py, pz = self.pos[p]
            old_vel = self.vel[p]

            new_vel = self._interp_vel(self.grid_u, self.grid_v, self.grid_w, px, py, pz)
            old_grid_vel = self._interp_vel(self.grid_u_old, self.grid_v_old, self.grid_w_old, px, py, pz)
            delta = new_vel - old_grid_vel

            pic_vel = new_vel
            flip_vel = old_vel + delta
            self.vel[p] = (1.0 - flip_ratio) * pic_vel + flip_ratio * flip_vel

    @ti.func
    def _interp_vel(self, u_field, v_field, w_field, px, py, pz) -> tm.vec3:
        """Interpolate full velocity at (px,py,pz) from staggered grid."""
        vx = self._interp_staggered(u_field, px / self.dx, py / self.dx - 0.5, pz / self.dx - 0.5,
                                     self.nx + 1, self.ny, self.nz)
        vy = self._interp_staggered(v_field, px / self.dx - 0.5, py / self.dx, pz / self.dx - 0.5,
                                     self.nx, self.ny + 1, self.nz)
        vz = self._interp_staggered(w_field, px / self.dx - 0.5, py / self.dx - 0.5, pz / self.dx,
                                     self.nx, self.ny, self.nz + 1)
        return tm.vec3(vx, vy, vz)

    @ti.func
    def _interp_staggered(self, field, gx, gy, gz, sx, sy, sz) -> float:
        """Trilinear interpolation on a staggered grid component.

        Args:
            gx, gy, gz: position in grid-index space (already offset for the component)
            sx, sy, sz: field shape dimensions for clamping
        """
        i0 = int(ti.floor(gx))
        j0 = int(ti.floor(gy))
        k0 = int(ti.floor(gz))
        fx = gx - i0
        fy = gy - j0
        fz = gz - k0

        result = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ni = ti.max(0, ti.min(sx - 1, i0 + di))
            nj = ti.max(0, ti.min(sy - 1, j0 + dj))
            nk = ti.max(0, ti.min(sz - 1, k0 + dk))
            w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
            result += field[ni, nj, nk] * w
        return result

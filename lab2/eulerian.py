"""Eulerian grid-based fluid simulator with semi-Lagrangian + AR advection."""
import taichi as ti
import taichi.math as tm
from lab2.core import FluidSimulator


@ti.data_oriented
class EulerianSimulator(FluidSimulator):
    """Pure grid-based Eulerian fluid simulator.

    Uses density-weighted gravity and semi-Lagrangian advection
    for velocity and density fields on a MAC grid.
    """

    def __init__(self, nx: int, ny: int, nz: int):
        num_particles = nx * ny * nz
        super().__init__(nx, ny, nz, num_particles=num_particles)

        self.density = ti.field(dtype=float, shape=(nx, ny, nz))
        self.density_new = ti.field(dtype=float, shape=(nx, ny, nz))
        self._d_tmp1 = ti.field(dtype=float, shape=(nx, ny, nz))
        self._d_tmp2 = ti.field(dtype=float, shape=(nx, ny, nz))

        self._render_density = ti.field(dtype=float, shape=num_particles)
        self.use_ar = False

    def substep(
        self,
        dt: float,
        flip_ratio: float = 0.0,
        gravity: float = -9.8,
        num_pressure_iters: int = 80,
        num_particle_iters: int = 0,
        over_relaxation: float = 1.9,
        compensate_drift: bool = False,
        separate_particles: bool = False,
    ):
        self.advect_velocity(dt)
        self.apply_gravity_density(dt, gravity)
        self.set_boundary_velocity()
        self.solve_incompressibility(num_pressure_iters, dt, over_relaxation, False)
        self.set_boundary_velocity()
        self.advect_density_sl(dt)
        self._rescale_density()
        self.density.copy_from(self.density_new)

    # ── Velocity advection ──

    @ti.kernel
    def advect_velocity(self, dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            x = i * dx
            y = (j + 0.5) * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_u[i, j, k] = self._interp_u(
                x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx
            )
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            x = (i + 0.5) * dx
            y = j * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_v[i, j, k] = self._interp_v(
                x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx
            )
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dx
            z = k * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_w[i, j, k] = self._interp_w(
                x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx
            )

    @ti.kernel
    def apply_gravity_density(self, dt: float, gravity: float):
        """Apply gravity weighted by density. Only fluid cells get gravity."""
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            # Average density at v-face (between cells j-1 and j)
            d = 0.0
            count = 0
            if j > 0:
                d += self.density[i, j - 1, k]
                count += 1
            if j < self.ny:
                d += self.density[i, j, k]
                count += 1
            if count > 0:
                d /= count
            if d > 0.01:
                self.grid_v[i, j, k] += gravity * dt

    # ── Density advection ──

    @ti.kernel
    def advect_density_sl(self, dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dx
            cz = (k + 0.5) * dx
            vel = self._interp_vel_self(cx, cy, cz, dx)
            self.density_new[i, j, k] = self._interp_density(
                cx - dt * vel[0], cy - dt * vel[1], cz - dt * vel[2], dx
            )

    @ti.kernel
    def advect_density_flux(self, dt: float):
        """Conservative upwind flux-based density advection (mass-preserving)."""
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            d = self.density[i, j, k]

            flux = 0.0

            # X-direction faces
            u_right = self.grid_u[i + 1, j, k]
            u_left = self.grid_u[i, j, k]
            if u_right > 0.0:
                flux -= d * u_right * dt / dx
            elif i < self.nx - 1:
                flux -= self.density[i + 1, j, k] * u_right * dt / dx
            if u_left > 0.0:
                if i > 0:
                    flux += self.density[i - 1, j, k] * u_left * dt / dx
            else:
                flux += d * u_left * dt / dx

            # Y-direction faces
            v_top = self.grid_v[i, j + 1, k]
            v_bot = self.grid_v[i, j, k]
            if v_top > 0.0:
                flux -= d * v_top * dt / dx
            elif j < self.ny - 1:
                flux -= self.density[i, j + 1, k] * v_top * dt / dx
            if v_bot > 0.0:
                if j > 0:
                    flux += self.density[i, j - 1, k] * v_bot * dt / dx
            else:
                flux += d * v_bot * dt / dx

            # Z-direction faces
            w_front = self.grid_w[i, j, k + 1]
            w_back = self.grid_w[i, j, k]
            if w_front > 0.0:
                flux -= d * w_front * dt / dx
            elif k < self.nz - 1:
                flux -= self.density[i, j, k + 1] * w_front * dt / dx
            if w_back > 0.0:
                if k > 0:
                    flux += self.density[i, j, k - 1] * w_back * dt / dx
            else:
                flux += d * w_back * dt / dx

            self.density_new[i, j, k] = d + flux
            if self.density_new[i, j, k] < 0.0:
                self.density_new[i, j, k] = 0.0
            if self.density_new[i, j, k] > 1.0:
                self.density_new[i, j, k] = 1.0

    @ti.kernel
    def _rescale_density(self):
        """Global mass conservation: rescale density_new to match density total mass."""
        total_old = 0.0
        total_new = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            total_old += self.density[i, j, k]
            total_new += self.density_new[i, j, k]

        if total_new > 1e-6:
            scale = total_old / total_new
            for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
                self.density_new[i, j, k] *= scale

    # ── Interpolation ──

    @ti.func
    def _interp_vel_self(self, px, py, pz, dx) -> tm.vec3:
        return tm.vec3(
            self._interp_u(px, py, pz, dx),
            self._interp_v(px, py, pz, dx),
            self._interp_w(px, py, pz, dx),
        )

    @ti.func
    def _interp_u(self, px, py, pz, dx) -> float:
        return self._interp_field(self.grid_u, px / dx, py / dx - 0.5, pz / dx - 0.5)

    @ti.func
    def _interp_v(self, px, py, pz, dx) -> float:
        return self._interp_field(self.grid_v, px / dx - 0.5, py / dx, pz / dx - 0.5)

    @ti.func
    def _interp_w(self, px, py, pz, dx) -> float:
        return self._interp_field(self.grid_w, px / dx - 0.5, py / dx - 0.5, pz / dx)

    @ti.func
    def _interp_density(self, px, py, pz, dx) -> float:
        return self._interp_field(self.density, px / dx - 0.5, py / dx - 0.5, pz / dx - 0.5)

    @ti.func
    def _interp_field(self, field, gx, gy, gz) -> float:
        i0 = ti.floor(gx, int)
        j0 = ti.floor(gy, int)
        k0 = ti.floor(gz, int)
        fx = gx - i0
        fy = gy - j0
        fz = gz - k0
        result = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ni = ti.max(0, ti.min(self.nx - 1, i0 + di))
            nj = ti.max(0, ti.min(self.ny - 1, j0 + dj))
            nk = ti.max(0, ti.min(self.nz - 1, k0 + dk))
            w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
            result += field[ni, nj, nk] * w
        return result

    # ── Cell type ──

    @ti.kernel
    def init_cell_types(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.cell_type[i, j, k] = 0 if self.density[i, j, k] > 0.01 else 1

    @ti.kernel
    def relabel_cells(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 2:
                self.cell_type[i, j, k] = 0 if self.density[i, j, k] > 0.01 else 1

    # ── Scene Init ──

    @ti.kernel
    def init_dam_break_density(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.density[i, j, k] = 0.0
            self.density_new[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            self.grid_u[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if i < self.nx // 2 and j < self.ny // 2 and k < self.nz // 2:
                self.density[i, j, k] = 1.0
                self.density_new[i, j, k] = 1.0

    # ── Render point update ──

    def update_render_points(self):
        self._build_render_points()

    @ti.kernel
    def _build_render_points(self):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            idx = i + j * self.nx + k * self.nx * self.ny
            d = self.density_new[i, j, k]
            if d > 0.01:
                self.pos[idx] = [(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx]
                prev_d = self._render_density[idx]
                smoothed_d = prev_d * 0.9 + d * 0.1
                if ti.abs(smoothed_d - prev_d) < 0.005:
                    smoothed_d = prev_d
                self._render_density[idx] = smoothed_d
                sd = smoothed_d
                self.color[idx] = [0.2 + 0.8 * sd, 0.5, 1.0 - sd * 0.5]
            else:
                self.pos[idx] = [0.5, -10.0, 0.5]
                self.color[idx] = [0.1, 0.1, 0.15]
                self._render_density[idx] = 0.0

"""Eulerian grid-based fluid simulator with semi-Lagrangian + AR advection."""
import taichi as ti
import taichi.math as tm
from lab2.core import FluidSimulator


@ti.data_oriented
class EulerianSimulator(FluidSimulator):
    """Pure grid-based Eulerian fluid simulator.

    Supports two advection modes for density:
      - Standard semi-Lagrangian (SL)
      - Advection-Reflection (AR) — reduces numerical diffusion

    Velocity is always advected via standard semi-Lagrangian.
    No particles — velocity and density live on the grid.
    """

    def __init__(self, nx: int, ny: int, nz: int):
        num_particles = nx * ny * nz
        super().__init__(nx, ny, nz, num_particles=num_particles)

        self.density = ti.field(dtype=float, shape=(nx, ny, nz))
        self.density_new = ti.field(dtype=float, shape=(nx, ny, nz))
        self._d_tmp1 = ti.field(dtype=float, shape=(nx, ny, nz))
        self._d_tmp2 = ti.field(dtype=float, shape=(nx, ny, nz))

        self._render_density = ti.field(dtype=float, shape=num_particles)
        self.use_ar = True

    def substep(
        self,
        dt: float,
        flip_ratio: float = 0.0,
        gravity: float = -9.8,
        num_pressure_iters: int = 40,
        num_particle_iters: int = 0,
        over_relaxation: float = 1.9,
        compensate_drift: bool = False,
        separate_particles: bool = False,
    ):
        self.advect_velocity(dt)
        self.apply_gravity(dt, gravity)
        self.set_boundary_velocity()
        self.relabel_cells()
        self.solve_incompressibility(num_pressure_iters, dt, over_relaxation, False)
        self.set_boundary_velocity()
        if self.use_ar:
            self._advect_density_ar(dt)
        else:
            self.advect_density_sl(dt)
        self.density.copy_from(self.density_new)

    # ── Velocity advection (standard semi-Lagrangian) ──

    @ti.kernel
    def advect_velocity(self, dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            x = i * dx
            y = (j + 0.5) * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_u[i, j, k] = self._interp_u(x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx)

        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            x = (i + 0.5) * dx
            y = j * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_v[i, j, k] = self._interp_v(x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx)

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dx
            z = k * dx
            vel = self._interp_vel_self(x, y, z, dx)
            self.grid_w[i, j, k] = self._interp_w(x - dt * vel[0], y - dt * vel[1], z - dt * vel[2], dx)

    @ti.kernel
    def apply_gravity(self, dt: float, gravity: float):
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] += gravity * dt

    # ── Density advection ──

    @ti.kernel
    def advect_density_sl(self, dt: float):
        """Standard semi-Lagrangian density advection (backward trace)."""
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dx
            cz = (k + 0.5) * dx
            vel = self._interp_vel_self(cx, cy, cz, dx)
            self.density_new[i, j, k] = self._interp_density(
                cx - dt * vel[0], cy - dt * vel[1], cz - dt * vel[2], dx
            )

    def _advect_density_ar(self, dt: float):
        """Advection-Reflection density advection.

        AR (Zehnder et al.) reduces numerical diffusion by:
          1. Standard backward advection:  d_back = SL(d, dt)
          2. Forward advection of result:   d_fwd = SL(d_back, -dt)
          3. Reflect the lost details:      d_new = d_back + (d - d_fwd)
        """
        self._advect_density_backward(self.density, self.density_new, dt)
        self._advect_density_forward(self.density_new, self._d_tmp1, dt)
        self._reflect_density(self.density, self.density_new, self._d_tmp1, self._d_tmp2)
        self.density_new.copy_from(self._d_tmp2)

    @ti.kernel
    def _advect_density_backward(self, src: ti.template(), dst: ti.template(), dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dx
            cz = (k + 0.5) * dx
            vel = self._interp_vel_self(cx, cy, cz, dx)
            dst[i, j, k] = self._interp_field(src, cx - dt * vel[0], cy - dt * vel[1], cz - dt * vel[2])

    @ti.kernel
    def _advect_density_forward(self, src: ti.template(), dst: ti.template(), dt: float):
        """Forward advection: trace from departure point forward to arrival."""
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dx
            cz = (k + 0.5) * dx
            vel = self._interp_vel_self(cx, cy, cz, dx)
            dst[i, j, k] = self._interp_field(src, cx + dt * vel[0], cy + dt * vel[1], cz + dt * vel[2])

    @ti.kernel
    def _reflect_density(self, d_orig: ti.template(), d_back: ti.template(),
                         d_fwd: ti.template(), d_out: ti.template()):
        """AR reflection: d_out = d_back + (d_orig - d_fwd), clamped to [0, 1]."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            correction = d_orig[i, j, k] - d_fwd[i, j, k]
            result = d_back[i, j, k] + correction
            if result < 0.0:
                result = 0.0
            if result > 1.0:
                result = 1.0
            d_out[i, j, k] = result

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
        """Interpolate a cell-centered field at fractional grid coordinates."""
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
            self.grid_u[i, j, k] = 0.0
            self.grid_w[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] = 0.0
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

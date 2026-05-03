import taichi as ti
import taichi.math as tm
from lab2.core import FluidSimulator


@ti.data_oriented
class EulerianSimulator(FluidSimulator):
    """Pure grid-based Eulerian fluid simulator with semi-Lagrangian advection.

    Operates on the same MAC staggered grid as FLIP.
    No particles — velocity and density are stored and advected on the grid.
    """

    def __init__(self, nx: int, ny: int, nz: int):
        super().__init__(nx, ny, nz, num_particles=nx * ny * nz)

        # Density field (for visualization)
        self.density = ti.field(dtype=float, shape=(nx, ny, nz))
        self.density_new = ti.field(dtype=float, shape=(nx, ny, nz))

        # Use particle fields as "render proxies" — one per cell center
        self.num_render = nx * ny * nz

    def substep(
        self,
        dt: float,
        flip_ratio: float = 0.0,  # unused in Eulerian
        gravity: float = -9.8,
        num_pressure_iters: int = 40,
        num_particle_iters: int = 0,
        over_relaxation: float = 1.9,
        compensate_drift: bool = False,
        separate_particles: bool = False,
    ):
        self.advect_velocity(dt)
        self.apply_gravity(dt, gravity)
        self.relabel_cells()
        self.solve_incompressibility(num_pressure_iters, dt, over_relaxation, False)
        self.set_boundary_velocity()
        self.advect_density(dt)

    # ---- Semi-Lagrangian Advection ----

    @ti.kernel
    def advect_velocity(self, dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            # u-face at (i*dx, (j+0.5)*dx, (k+0.5)*dx)
            x = i * dx
            y = (j + 0.5) * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            px = x - dt * vel[0]
            py = y - dt * vel[1]
            pz = z - dt * vel[2]
            self.grid_u[i, j, k] = self._interp_u(px, py, pz, dx)

        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            # v-face at ((i+0.5)*dx, j*dx, (k+0.5)*dx)
            x = (i + 0.5) * dx
            y = j * dx
            z = (k + 0.5) * dx
            vel = self._interp_vel_self(x, y, z, dx)
            px = x - dt * vel[0]
            py = y - dt * vel[1]
            pz = z - dt * vel[2]
            self.grid_v[i, j, k] = self._interp_v(px, py, pz, dx)

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            # w-face at ((i+0.5)*dx, (j+0.5)*dx, k*dx)
            x = (i + 0.5) * dx
            y = (j + 0.5) * dx
            z = k * dx
            vel = self._interp_vel_self(x, y, z, dx)
            px = x - dt * vel[0]
            py = y - dt * vel[1]
            pz = z - dt * vel[2]
            self.grid_w[i, j, k] = self._interp_w(px, py, pz, dx)

    @ti.kernel
    def advect_density(self, dt: float):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dx
            cz = (k + 0.5) * dx
            vel = self._interp_vel_self(cx, cy, cz, dx)
            px = cx - dt * vel[0]
            py = cy - dt * vel[1]
            pz = cz - dt * vel[2]
            self.density[i, j, k] = self._interp_density(px, py, pz, dx)

    @ti.kernel
    def apply_gravity(self, dt: float, gravity: float):
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] += gravity * dt

    # ---- Interpolation helpers ----

    @ti.func
    def _interp_vel_self(self, px, py, pz, dx) -> tm.vec3:
        vx = self._interp_u(px, py, pz, dx)
        vy = self._interp_v(px, py, pz, dx)
        vz = self._interp_w(px, py, pz, dx)
        return tm.vec3(vx, vy, vz)

    @ti.func
    def _interp_u(self, px, py, pz, dx) -> float:
        return self._interp(self.grid_u, px / dx, py / dx - 0.5, pz / dx - 0.5,
                           self.nx + 1, self.ny, self.nz)

    @ti.func
    def _interp_v(self, px, py, pz, dx) -> float:
        return self._interp(self.grid_v, px / dx - 0.5, py / dx, pz / dx - 0.5,
                           self.nx, self.ny + 1, self.nz)

    @ti.func
    def _interp_w(self, px, py, pz, dx) -> float:
        return self._interp(self.grid_w, px / dx - 0.5, py / dx - 0.5, pz / dx,
                           self.nx, self.ny, self.nz + 1)

    @ti.func
    def _interp_density(self, px, py, pz, dx) -> float:
        return self._interp(self.density, px / dx - 0.5, py / dx - 0.5, pz / dx - 0.5,
                           self.nx, self.ny, self.nz)

    @ti.func
    def _interp(self, field, gx, gy, gz, sx, sy, sz) -> float:
        i0 = ti.floor(gx, int)
        j0 = ti.floor(gy, int)
        k0 = ti.floor(gz, int)
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

    # ---- Cell type from density ----

    @ti.kernel
    def init_cell_types(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.cell_type[i, j, k] = 0 if self.density[i, j, k] > 0.01 else 1

    @ti.kernel
    def relabel_cells(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 2:
                self.cell_type[i, j, k] = 0 if self.density[i, j, k] > 0.01 else 1

    # ---- Scene Init ----

    @ti.kernel
    def init_dam_break_density(self):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.density[i, j, k] = 0.0
            self.grid_u[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w[i, j, k] = 0.0
        # Dam break: fill left-front corner with density
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if i < self.nx // 2 and j < self.ny * 3 // 4 and k < self.nz // 2:
                self.density[i, j, k] = 1.0

    # ---- Density-based particle updates (for GGUI rendering) ----

    def update_render_points(self):
        self._build_render_points()

    @ti.kernel
    def _build_render_points(self):
        dx = self.dx
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            idx = i + j * self.nx + k * self.nx * self.ny
            d = self.density[i, j, k]
            if d > 0.01:
                self.pos[idx] = [
                    (i + 0.5) * dx,
                    (j + 0.5) * dx,
                    (k + 0.5) * dx,
                ]
                self.color[idx] = [0.2 + 0.8 * d, 0.5, 1.0 - d * 0.5]
            else:
                # Move far below domain and use background color so invisible
                self.pos[idx] = [0.5, -10.0, 0.5]
                self.color[idx] = [0.1, 0.1, 0.15]

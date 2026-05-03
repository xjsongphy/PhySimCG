import taichi as ti
import taichi.math as tm


def scene_particle_count(scene_name: str, nx: int) -> int:
    """Compute exact particle count for a scene at given grid resolution."""
    dx = 1.0 / nx
    spacing = dx * 0.55

    if scene_name == "Dam Break":
        npx = int((0.45 - 0.05) / spacing)
        npy = int((0.85 - 0.05) / spacing)
        npz = int((0.45 - 0.05) / spacing)
        return npx * npy * npz

    elif scene_name == "Drop":
        cx, cy, cz = 0.5, 0.75, 0.5
        r = 0.15
        npx = int((2 * r) / spacing)
        npy = int((2 * r) / spacing)
        npz = int((2 * r) / spacing)
        return npx * npy * npz

    elif scene_name == "Double Dam":
        npx1 = int((0.25 - 0.05) / spacing)
        npy1 = int((0.65 - 0.05) / spacing)
        npz1 = int((0.45 - 0.05) / spacing)
        npx2 = int((0.95 - 0.75) / spacing)
        npy2 = int((0.65 - 0.05) / spacing)
        npz2 = int((0.95 - 0.55) / spacing)
        return npx1 * npy1 * npz1 + npx2 * npy2 * npz2

    return 0


@ti.data_oriented
class FluidSimulator:
    """Shared data structures and kernels for all fluid simulation methods.

    Staggered grid (MAC) convention:
    - Cell (i,j,k) spans [i*dx, (i+1)*dx] x [j*dx, (j+1)*dx] x [k*dx, (k+1)*dx]
    - Cell center at ((i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx)
    - u[i,j,k] at (i*dx, (j+0.5)*dx, (k+0.5)*dx)  — left face of cell
    - v[i,j,k] at ((i+0.5)*dx, j*dx, (k+0.5)*dx)  — bottom face of cell
    - w[i,j,k] at ((i+0.5)*dx, (j+0.5)*dx, k*dx)  — front face of cell
    - Cell has u on left face (i) and right face (i+1)
    - Cell has v on bottom face (j) and top face (j+1)
    - Cell has w on front face (k) and back face (k+1)
    """

    def __init__(self, nx: int, ny: int, nz: int, num_particles: int):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.num_particles = num_particles
        self.dx = 1.0 / nx

        # Particle data
        self.pos = ti.Vector.field(3, dtype=float, shape=num_particles)
        self.vel = ti.Vector.field(3, dtype=float, shape=num_particles)
        self.color = ti.Vector.field(3, dtype=float, shape=num_particles)

        # Staggered grid (MAC): velocity at face centers
        self.grid_u = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w = ti.field(dtype=float, shape=(nx, ny, nz + 1))
        self.grid_u_old = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v_old = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w_old = ti.field(dtype=float, shape=(nx, ny, nz + 1))

        # Cell type: 0=fluid, 1=air, 2=solid
        self.cell_type = ti.field(dtype=int, shape=(nx, ny, nz))

        self.particle_density = ti.field(dtype=float, shape=(nx, ny, nz))
        self.particle_density_init = ti.field(dtype=float, shape=(nx, ny, nz))

        self.grid_u_weight = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v_weight = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w_weight = ti.field(dtype=float, shape=(nx, ny, nz + 1))

        # Neighbor grid for particle separation
        self._max_per_cell = 20
        self.grid_particle_count = ti.field(dtype=int, shape=(nx, ny, nz))
        self.grid_particle_ids = ti.field(dtype=int, shape=(nx, ny, nz, self._max_per_cell))

        # Obstacle (sphere)
        self.obstacle_pos = ti.Vector.field(3, dtype=float, shape=(1,))
        self.obstacle_vel = ti.Vector.field(3, dtype=float, shape=(1,))
        self.obstacle_radius = ti.field(dtype=float, shape=())
        self.obstacle_pos[0] = [0.5, 0.5, 0.5]
        self.obstacle_radius[None] = 0.0

    # ---- Scene Initialization ----

    @ti.kernel
    def init_dam_break(self):
        dx = self.dx
        spacing = dx * 0.55
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        npx = int((hi_x - lo_x) / spacing)
        npy = int((hi_y - lo_y) / spacing)
        npz = int((hi_z - lo_z) / spacing)
        for i in range(self.num_particles):
            if i < npx * npy * npz:
                ix = i % npx
                iy = (i // npx) % npy
                iz = i // (npx * npy)
                self.pos[i] = [
                    lo_x + (ix + 0.5) * spacing,
                    lo_y + (iy + 0.5) * spacing,
                    lo_z + (iz + 0.5) * spacing,
                ]
                self.vel[i] = [0.0, 0.0, 0.0]
            else:
                self.pos[i] = [-100.0, -100.0, -100.0]
                self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_drop(self):
        dx = self.dx
        spacing = dx * 0.55
        cx, cy, cz = 0.5, 0.75, 0.5
        radius = 0.15
        lo_x = cx - radius
        hi_x = cx + radius
        lo_y = cy - radius
        hi_y = cy + radius
        lo_z = cz - radius
        hi_z = cz + radius
        npx = int((hi_x - lo_x) / spacing)
        npy = int((hi_y - lo_y) / spacing)
        npz = int((hi_z - lo_z) / spacing)
        for i in range(self.num_particles):
            ix = i % npx
            iy = (i // npx) % npy
            iz = i // (npx * npy)
            self.pos[i] = [
                lo_x + (ix + 0.5) * spacing,
                lo_y + (iy + 0.5) * spacing,
                lo_z + (iz + 0.5) * spacing,
            ]
            self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_double_dam(self):
        dx = self.dx
        spacing = dx * 0.55
        lo_x1, hi_x1 = 0.05, 0.25
        lo_y1, hi_y1 = 0.05, 0.65
        lo_z1, hi_z1 = 0.05, 0.45
        npx1 = int((hi_x1 - lo_x1) / spacing)
        npy1 = int((hi_y1 - lo_y1) / spacing)
        npz1 = int((hi_z1 - lo_z1) / spacing)
        count1 = npx1 * npy1 * npz1
        lo_x2, hi_x2 = 0.75, 0.95
        lo_y2, hi_y2 = 0.05, 0.65
        lo_z2, hi_z2 = 0.55, 0.95
        npx2 = int((hi_x2 - lo_x2) / spacing)
        npy2 = int((hi_y2 - lo_y2) / spacing)
        npz2 = int((hi_z2 - lo_z2) / spacing)
        for i in range(self.num_particles):
            if i < count1:
                ix = i % npx1
                iy = (i // npx1) % npy1
                iz = i // (npx1 * npy1)
                self.pos[i] = [
                    lo_x1 + (ix + 0.5) * spacing,
                    lo_y1 + (iy + 0.5) * spacing,
                    lo_z1 + (iz + 0.5) * spacing,
                ]
                self.vel[i] = [0.0, 0.0, 0.0]
            elif i < count1 + npx2 * npy2 * npz2:
                idx = i - count1
                ix = idx % npx2
                iy = (idx // npx2) % npy2
                iz = idx // (npx2 * npy2)
                self.pos[i] = [
                    lo_x2 + (ix + 0.5) * spacing,
                    lo_y2 + (iy + 0.5) * spacing,
                    lo_z2 + (iz + 0.5) * spacing,
                ]
                self.vel[i] = [0.0, 0.0, 0.0]
            else:
                self.pos[i] = [-100.0, -100.0, -100.0]
                self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_colors(self):
        for i in range(self.num_particles):
            if self.pos[i][0] >= 0:
                self.color[i] = [0.2, 0.5, 1.0]
            else:
                self.color[i] = [0.1, 0.1, 0.15]

    # ---- Cell Type Management ----

    @ti.kernel
    def init_cell_types(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.cell_type[i, j, k] = 1  # air
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / self.dx)
            cj = int(self.pos[p][1] / self.dx)
            ck = int(self.pos[p][2] / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                self.cell_type[ci, cj, ck] = 0  # fluid

    @ti.kernel
    def mark_obstacle_cells(self):
        obs_pos = self.obstacle_pos[0]
        obs_r = self.obstacle_radius[None]
        dx = self.dx
        if obs_r > 0:
            for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dx
                cz = (k + 0.5) * dx
                dist2 = (cx - obs_pos[0]) ** 2 + (cy - obs_pos[1]) ** 2 + (cz - obs_pos[2]) ** 2
                if dist2 < (obs_r + dx) ** 2:
                    self.cell_type[i, j, k] = 2

    @ti.kernel
    def clear_obstacle_cells(self):
        """Reset only obstacle solid cells back to air."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 2:
                self.cell_type[i, j, k] = 1
        # Re-mark fluid cells from particles (only air cells)
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / self.dx)
            cj = int(self.pos[p][1] / self.dx)
            ck = int(self.pos[p][2] / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                if self.cell_type[ci, cj, ck] == 1:
                    self.cell_type[ci, cj, ck] = 0

    @ti.kernel
    def relabel_cells(self):
        """Relabel fluid/air cells based on current particle positions.
        Solid (obstacle) cells are preserved."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 2:
                self.cell_type[i, j, k] = 1  # air
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / self.dx)
            cj = int(self.pos[p][1] / self.dx)
            ck = int(self.pos[p][2] / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                self.cell_type[ci, cj, ck] = 0  # fluid

    # ---- Simulation Kernels ----

    @ti.kernel
    def integrate_particles(self, dt: float, gravity: float):
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            self.vel[i][1] += gravity * dt
            self.pos[i] += self.vel[i] * dt

    @ti.kernel
    def handle_particle_collisions(self):
        """Clamp particles to domain [eps, 1-eps], zero out-boundary velocity."""
        eps = 1e-6
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            for d in ti.static(range(3)):
                if self.pos[i][d] < eps:
                    self.pos[i][d] = eps
                    if self.vel[i][d] < 0:
                        self.vel[i][d] = 0.0
                elif self.pos[i][d] > 1.0 - eps:
                    self.pos[i][d] = 1.0 - eps
                    if self.vel[i][d] > 0:
                        self.vel[i][d] = 0.0
            # Sphere obstacle collision
            obs_pos = self.obstacle_pos[0]
            obs_r = self.obstacle_radius[None]
            if obs_r > 0:
                diff = self.pos[i] - obs_pos
                dist = diff.norm()
                if dist < obs_r and dist > 1e-8:
                    n = diff / dist
                    self.pos[i] = obs_pos + n * obs_r
                    obs_vel = self.obstacle_vel[0]
                    rel_vel = self.vel[i] - obs_vel
                    vn = rel_vel.dot(n)
                    if vn < 0:
                        self.vel[i] = self.vel[i] - n * vn + obs_vel * 0.5

    def push_particles_apart(self, num_iters: int):
        for _ in range(num_iters):
            self._clear_neighbor_grid()
            self._build_neighbor_grid()
            self._separate_pass()

    @ti.kernel
    def _clear_neighbor_grid(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.grid_particle_count[i, j, k] = 0

    @ti.kernel
    def _build_neighbor_grid(self):
        dx = self.dx
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / dx)
            cj = int(self.pos[p][1] / dx)
            ck = int(self.pos[p][2] / dx)
            ci = ti.max(0, ti.min(self.nx - 1, ci))
            cj = ti.max(0, ti.min(self.ny - 1, cj))
            ck = ti.max(0, ti.min(self.nz - 1, ck))
            slot = ti.atomic_add(self.grid_particle_count[ci, cj, ck], 1)
            if slot < self._max_per_cell:
                self.grid_particle_ids[ci, cj, ck, slot] = p

    @ti.kernel
    def _separate_pass(self):
        dx = self.dx
        min_dist = dx * 0.6
        min_dist2 = min_dist * min_dist
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / dx)
            cj = int(self.pos[p][1] / dx)
            ck = int(self.pos[p][2] / dx)
            for di, dj, dk in ti.static(ti.ndrange(3, 3, 3)):
                ni = ci + di - 1
                nj = cj + dj - 1
                nk = ck + dk - 1
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    cnt = self.grid_particle_count[ni, nj, nk]
                    for s in range(ti.i32(self._max_per_cell)):
                        if s >= cnt:
                            continue
                        q = self.grid_particle_ids[ni, nj, nk, s]
                        if q != p:
                            diff = self.pos[p] - self.pos[q]
                            d2 = diff.dot(diff)
                            if d2 < min_dist2 and d2 > 1e-12:
                                d = ti.sqrt(d2)
                                n_dir = diff / d
                                self.pos[p] += n_dir * (min_dist - d) * 0.7

    @ti.kernel
    def update_particle_density(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.particle_density[i, j, k] = 0.0
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / self.dx)
            cj = int(self.pos[p][1] / self.dx)
            ck = int(self.pos[p][2] / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                self.particle_density[ci, cj, ck] += 1.0

    @ti.kernel
    def store_initial_density(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.particle_density_init[i, j, k] = self.particle_density[i, j, k]

    @ti.kernel
    def apply_perturbation(self, center_x: float, center_y: float, center_z: float,
                           strength: float):
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            diff = self.pos[i] - tm.vec3([center_x, center_y, center_z])
            dist = diff.norm()
            if 0.01 < dist < 0.3:
                n = diff / dist
                self.vel[i] += n * strength * (0.3 - dist) / 0.3

    # ---- Gauss-Seidel Pressure Projection (Red-Black) ----

    def solve_incompressibility(
        self,
        num_iters: int,
        dt: float,
        over_relaxation: float,
        compensate_drift: bool,
    ):
        for _ in range(num_iters):
            self._gs_pass(0, over_relaxation, compensate_drift)
            self._gs_pass(1, over_relaxation, compensate_drift)

    @ti.kernel
    def _gs_pass(self, color: int, over_relaxation: float, compensate_drift: bool):
        """One sweep of Red-Black Gauss-Seidel for the given color."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if (i + j + k) % 2 != color:
                continue
            if self.cell_type[i, j, k] != 0:
                continue

            div = (
                self.grid_u[i + 1, j, k]
                - self.grid_u[i, j, k]
                + self.grid_v[i, j + 1, k]
                - self.grid_v[i, j, k]
                + self.grid_w[i, j, k + 1]
                - self.grid_w[i, j, k]
            )

            s = 0.0
            if i > 0 and self.cell_type[i - 1, j, k] != 2:
                s += 1.0
            if i < self.nx - 1 and self.cell_type[i + 1, j, k] != 2:
                s += 1.0
            if j > 0 and self.cell_type[i, j - 1, k] != 2:
                s += 1.0
            if j < self.ny - 1 and self.cell_type[i, j + 1, k] != 2:
                s += 1.0
            if k > 0 and self.cell_type[i, j, k - 1] != 2:
                s += 1.0
            if k < self.nz - 1 and self.cell_type[i, j, k + 1] != 2:
                s += 1.0

            if s < 0.5:
                continue

            if compensate_drift:
                density_diff = (
                    self.particle_density[i, j, k]
                    - self.particle_density_init[i, j, k]
                )
                div -= density_diff * 0.1

            correction = over_relaxation * div / s

            if i > 0 and self.cell_type[i - 1, j, k] != 2:
                self.grid_u[i, j, k] += correction
            if i < self.nx - 1 and self.cell_type[i + 1, j, k] != 2:
                self.grid_u[i + 1, j, k] -= correction
            if j > 0 and self.cell_type[i, j - 1, k] != 2:
                self.grid_v[i, j, k] += correction
            if j < self.ny - 1 and self.cell_type[i, j + 1, k] != 2:
                self.grid_v[i, j + 1, k] -= correction
            if k > 0 and self.cell_type[i, j, k - 1] != 2:
                self.grid_w[i, j, k] += correction
            if k < self.nz - 1 and self.cell_type[i, j, k + 1] != 2:
                self.grid_w[i, j, k + 1] -= correction

    # ---- Grid Utilities ----

    @ti.kernel
    def clear_grid(self):
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            self.grid_u[i, j, k] = 0.0
            self.grid_u_weight[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v[i, j, k] = 0.0
            self.grid_v_weight[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w[i, j, k] = 0.0
            self.grid_w_weight[i, j, k] = 0.0

    @ti.kernel
    def save_grid_vel_old(self):
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            self.grid_u_old[i, j, k] = self.grid_u[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v_old[i, j, k] = self.grid_v[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w_old[i, j, k] = self.grid_w[i, j, k]

    @ti.kernel
    def normalize_grid_vel(self):
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            w = self.grid_u_weight[i, j, k]
            if w > 1e-8:
                self.grid_u[i, j, k] /= w
            else:
                self.grid_u[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            w = self.grid_v_weight[i, j, k]
            if w > 1e-8:
                self.grid_v[i, j, k] /= w
            else:
                self.grid_v[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            w = self.grid_w_weight[i, j, k]
            if w > 1e-8:
                self.grid_w[i, j, k] /= w
            else:
                self.grid_w[i, j, k] = 0.0

    @ti.kernel
    def set_boundary_velocity(self):
        for j, k in ti.ndrange(self.ny, self.nz):
            self.grid_u[0, j, k] = 0.0
            self.grid_u[self.nx, j, k] = 0.0
        for i, k in ti.ndrange(self.nx, self.nz):
            self.grid_v[i, 0, k] = 0.0
            self.grid_v[i, self.ny, k] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            self.grid_w[i, j, 0] = 0.0
            self.grid_w[i, j, self.nz] = 0.0

    # ---- Coloring ----

    @ti.kernel
    def update_default_colors(self):
        for i in range(self.num_particles):
            if self.pos[i][0] >= 0:
                speed = self.vel[i].norm()
                t = ti.min(speed / 5.0, 1.0)
                self.color[i] = [0.2 + 0.8 * t, 0.5 + 0.5 * t, 1.0]
            else:
                self.color[i] = [0.1, 0.1, 0.15]

    @ti.kernel
    def update_colors_by_density(self):
        for i in range(self.num_particles):
            if self.pos[i][0] >= 0:
                ci = int(self.pos[i][0] / self.dx)
                cj = int(self.pos[i][1] / self.dx)
                ck = int(self.pos[i][2] / self.dx)
                if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                    d = self.particle_density[ci, cj, ck]
                    t = ti.min(d / 8.0, 1.0)
                    self.color[i] = [t, 0.3 * (1.0 - t), 1.0 - t]
                else:
                    self.color[i] = [0.2, 0.5, 1.0]
            else:
                self.color[i] = [0.1, 0.1, 0.15]

    @ti.kernel
    def update_colors_uniform(self):
        for i in range(self.num_particles):
            if self.pos[i][0] >= 0:
                self.color[i] = [0.2, 0.5, 1.0]
            else:
                self.color[i] = [0.1, 0.1, 0.15]

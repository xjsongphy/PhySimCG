import taichi as ti
import taichi.math as tm


@ti.data_oriented
class FluidSimulator:
    """Shared data structures and kernels for all fluid simulation methods."""

    def __init__(self, nx: int, ny: int, nz: int, num_particles: int):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.num_particles = num_particles

        # Particle data
        self.pos = ti.Vector.field(3, dtype=float, shape=num_particles)
        self.vel = ti.Vector.field(3, dtype=float, shape=num_particles)
        self.color = ti.Vector.field(3, dtype=float, shape=num_particles)

        # Staggered grid (MAC): velocity at face centers
        # u on x-faces: (nx+1, ny, nz)
        # v on y-faces: (nx, ny+1, nz)
        # w on z-faces: (nx, ny, nz+1)
        self.grid_u = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w = ti.field(dtype=float, shape=(nx, ny, nz + 1))
        # Old grid velocity (for FLIP delta)
        self.grid_u_old = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v_old = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w_old = ti.field(dtype=float, shape=(nx, ny, nz + 1))

        # Pressure at cell centers
        self.pressure = ti.field(dtype=float, shape=(nx, ny, nz))

        # Cell type: 0=fluid, 1=air, 2=solid
        self.cell_type = ti.field(dtype=int, shape=(nx, ny, nz))

        # Particle density per cell (for drift compensation)
        self.particle_density = ti.field(dtype=float, shape=(nx, ny, nz))
        self.particle_density_init = ti.field(dtype=float, shape=(nx, ny, nz))

        # Grid weight accumulators (for P2G normalization)
        self.grid_u_weight = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v_weight = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w_weight = ti.field(dtype=float, shape=(nx, ny, nz + 1))

        # Obstacle (sphere)
        self.obstacle_pos = ti.Vector.field(3, dtype=float, shape=())
        self.obstacle_vel = ti.Vector.field(3, dtype=float, shape=())
        self.obstacle_radius = ti.field(dtype=float, shape=())
        self.obstacle_pos[None] = [0.5, 0.5, 0.5]
        self.obstacle_radius[None] = 0.0

        # Domain size (assumed [0,1]^3, cell size = 1/nx)
        self.dx = 1.0 / nx

    @ti.kernel
    def init_dam_break(self):
        """Initialize dam break: fluid block in one corner."""
        dx = self.dx
        # Place particles in a block [0.05, 0.45] x [0.05, 0.85] x [0.05, 0.45]
        # Particle spacing = dx * 0.5 (2 particles per cell diameter)
        spacing = dx * 0.5
        idx = 0
        for i in range(self.num_particles):
            pass  # will fill below

        # Use a sequential approach: compute 3D index from particle index
        particles_per_axis = 0
        total = 0
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        nx_p = int((hi_x - lo_x) / spacing)
        ny_p = int((hi_y - lo_y) / spacing)
        nz_p = int((hi_z - lo_z) / spacing)

        for i in range(self.num_particles):
            if i < nx_p * ny_p * nz_p:
                ix = i % nx_p
                iy = (i // nx_p) % ny_p
                iz = i // (nx_p * ny_p)
                x = lo_x + (ix + 0.5) * spacing
                y = lo_y + (iy + 0.5) * spacing
                z = lo_z + (iz + 0.5) * spacing
                self.pos[i] = [x, y, z]
                self.vel[i] = [0.0, 0.0, 0.0]
            else:
                # Extra particles: place at origin (inactive)
                self.pos[i] = [-1.0, -1.0, -1.0]
                self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_cell_types(self):
        """Mark boundary cells as solid, interior based on particles."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.cell_type[i, j, k] = 1  # air by default

        # Mark cells containing particles as fluid
        for p in range(self.num_particles):
            px, py, pz = self.pos[p]
            if px < 0 or py < 0 or pz < 0:
                continue
            ci = int(px / self.dx)
            cj = int(py / self.dx)
            ck = int(pz / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                self.cell_type[ci, cj, ck] = 0  # fluid

    @ti.kernel
    def integrate_particles(self, dt: float, gravity: float):
        """Apply gravity and update particle positions."""
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            self.vel[i][1] += gravity * dt
            self.pos[i] += self.vel[i] * dt

    @ti.kernel
    def handle_particle_collisions(self):
        """Handle boundary collisions and sphere obstacle."""
        dx = self.dx
        restitution = 0.0
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue

            # Domain boundaries [dx, 1-dx]
            for d in ti.static(range(3)):
                if self.pos[i][d] < dx:
                    self.pos[i][d] = dx
                    if self.vel[i][d] < 0:
                        self.vel[i][d] *= -restitution
                elif self.pos[i][d] > 1.0 - dx:
                    self.pos[i][d] = 1.0 - dx
                    if self.vel[i][d] > 0:
                        self.vel[i][d] *= -restitution

            # Sphere obstacle collision
            obs_pos = self.obstacle_pos[None]
            obs_r = self.obstacle_radius[None]
            if obs_r > 0:
                diff = self.pos[i] - obs_pos
                dist = diff.norm()
                if dist < obs_r and dist > 1e-8:
                    n = diff / dist
                    self.pos[i] = obs_pos + n * obs_r
                    vn = self.vel[i].dot(n)
                    if vn < 0:
                        self.vel[i] = self.vel[i] - n * vn

    @ti.kernel
    def push_particles_apart(self, num_iters: int):
        """Separate overlapping particles using spatial hashing."""
        dx = self.dx
        min_dist = dx * 0.5  # particles should be at least half cell apart
        min_dist2 = min_dist * min_dist

        # Spatial hash grid
        grid_res = int(1.0 / min_dist) + 1

        for iter in range(num_iters):
            # Simple O(n^2) for now — spatial hash acceleration can be added later
            for i in range(self.num_particles):
                if self.pos[i][0] < 0:
                    continue
                for j in range(i + 1, self.num_particles):
                    if self.pos[j][0] < 0:
                        continue
                    diff = self.pos[i] - self.pos[j]
                    d2 = diff.dot(diff)
                    if d2 < min_dist2 and d2 > 1e-12:
                        d = ti.sqrt(d2)
                        n = diff / d
                        correction = n * (min_dist - d) * 0.5
                        self.pos[i] += correction
                        self.pos[j] -= correction

    @ti.kernel
    def update_particle_density(self):
        """Count particles per cell to compute density."""
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
        """Save initial particle density as reference for drift compensation."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.particle_density_init[i, j, k] = self.particle_density[i, j, k]

    @ti.kernel
    def solve_incompressibility(
        self,
        num_iters: int,
        dt: float,
        over_relaxation: float,
        compensate_drift: bool,
    ):
        """Gauss-Seidel pressure projection on staggered grid."""
        dx = self.dx
        scale = over_relaxation

        for iter in range(num_iters):
            for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
                if self.cell_type[i, j, k] != 0:
                    continue

                # Compute divergence from staggered velocity
                div = (
                    self.grid_u[i + 1, j, k]
                    - self.grid_u[i, j, k]
                    + self.grid_v[i, j + 1, k]
                    - self.grid_v[i, j, k]
                    + self.grid_w[i, j, k + 1]
                    - self.grid_w[i, j, k]
                )

                # Count fluid/air neighbors for normalization
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

                if s < 1e-6:
                    continue

                # Drift compensation
                if compensate_drift:
                    density_diff = (
                        self.particle_density[i, j, k]
                        - self.particle_density_init[i, j, k]
                    )
                    div -= density_diff * 0.1

                correction = scale * div / s

                # Correct staggered velocities
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

    @ti.kernel
    def clear_grid(self):
        """Zero out grid velocity and weight accumulators."""
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
        """Save current grid velocity before pressure solve (for FLIP)."""
        for i, j, k in ti.ndrange(self.nx + 1, self.ny, self.nz):
            self.grid_u_old[i, j, k] = self.grid_u[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny + 1, self.nz):
            self.grid_v_old[i, j, k] = self.grid_v[i, j, k]
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz + 1):
            self.grid_w_old[i, j, k] = self.grid_w[i, j, k]

    @ti.kernel
    def normalize_grid_vel(self):
        """Divide accumulated P2G velocity by weights."""
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
    def update_default_colors(self):
        """Set default blue color for all active particles."""
        for i in range(self.num_particles):
            if self.pos[i][0] >= 0:
                speed = self.vel[i].norm()
                # Color by speed: blue (slow) -> cyan -> white (fast)
                t = ti.min(speed / 5.0, 1.0)
                self.color[i] = [0.2 + 0.8 * t, 0.5 + 0.5 * t, 1.0]
            else:
                self.color[i] = [0, 0, 0]

    @ti.kernel
    def set_boundary_velocity(self):
        """Set solid boundary velocity to zero."""
        # Left/right walls (x faces)
        for j, k in ti.ndrange(self.ny, self.nz):
            self.grid_u[0, j, k] = 0.0
            self.grid_u[self.nx, j, k] = 0.0
        # Bottom/top walls (y faces)
        for i, k in ti.ndrange(self.nx, self.nz):
            self.grid_v[i, 0, k] = 0.0
            self.grid_v[i, self.ny, k] = 0.0
        # Front/back walls (z faces)
        for i, j in ti.ndrange(self.nx, self.ny):
            self.grid_w[i, j, 0] = 0.0
            self.grid_w[i, j, self.nz] = 0.0

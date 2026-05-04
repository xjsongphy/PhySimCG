import taichi as ti
import taichi.math as tm


# Particle spacing as fraction of grid cell size (smaller = more particles, better packing)
_PARTICLE_SPACING = 0.7

def scene_particle_count(scene_name: str, nx: int) -> int:
    """Compute exact particle count for a scene at given grid resolution."""
    dx = 1.0 / nx
    sp = dx * _PARTICLE_SPACING

    if scene_name == "Dam Break":
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        npx = int((hi_x - lo_x) / sp)
        npy = int((hi_y - lo_y) / sp)
        npz = int((hi_z - lo_z) / sp)
        return npx * npy * npz

    elif scene_name == "Drop":
        cx, cy, cz = 0.5, 0.75, 0.5
        r = 0.15
        lo_x = cx - r
        hi_x = cx + r
        lo_y = cy - r
        hi_y = cy + r
        lo_z = cz - r
        hi_z = cz + r
        npx = int((hi_x - lo_x) / sp)
        npy = int((hi_y - lo_y) / sp)
        npz = int((hi_z - lo_z) / sp)
        return npx * npy * npz

    elif scene_name == "Double Dam":
        npx1 = int((0.25 - 0.05) / sp)
        npy1 = int((0.65 - 0.05) / sp)
        npz1 = int((0.45 - 0.05) / sp)
        npx2 = int((0.95 - 0.75) / sp)
        npy2 = int((0.65 - 0.05) / sp)
        npz2 = int((0.95 - 0.55) / sp)
        return npx1 * npy1 * npz1 + npx2 * npy2 * npz2

    elif scene_name == "Dam Break with Obstacle":
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        box_x = hi_x - lo_x
        box_y = hi_y - lo_y
        box_z = hi_z - lo_z
        npx = int(box_x / sp)
        npy = int(box_y / sp)
        npz = int(box_z / sp)
        return npx * npy * npz

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
        self._target_density = ti.field(dtype=float, shape=())

        self.grid_u_weight = ti.field(dtype=float, shape=(nx + 1, ny, nz))
        self.grid_v_weight = ti.field(dtype=float, shape=(nx, ny + 1, nz))
        self.grid_w_weight = ti.field(dtype=float, shape=(nx, ny, nz + 1))

        # Neighbor grid for particle separation
        self._max_per_cell = 8
        self.grid_particle_count = ti.field(dtype=int, shape=(nx, ny, nz))
        self.grid_particle_ids = ti.field(dtype=int, shape=(nx, ny, nz, self._max_per_cell))

        # CG pressure solver fields
        self.pressure = ti.field(dtype=float, shape=(nx, ny, nz))
        self._cg_r = ti.field(dtype=float, shape=(nx, ny, nz))
        self._cg_p = ti.field(dtype=float, shape=(nx, ny, nz))
        self._cg_Ap = ti.field(dtype=float, shape=(nx, ny, nz))
        self._cg_rdotr = ti.field(dtype=float, shape=(1,))
        self._cg_pAp = ti.field(dtype=float, shape=(1,))

        # Debug: stuck particle tracking
        self._stuck_count = ti.field(dtype=int, shape=(1,))
        self._prev_pos = ti.Vector.field(3, dtype=float, shape=num_particles)

        # Volume tracking
        self._fluid_cell_count = ti.field(dtype=int, shape=(1,))
        self._init_fluid_cells = ti.field(dtype=int, shape=(1,))
        self._fluid_vol_ratio = ti.field(dtype=float, shape=(1,))

        # Color mode flags
        self._color_density_updated = False

        # Obstacles (up to 4 spheres/boxes)
        self._max_obstacles = 4
        self.obstacle_count = ti.field(dtype=int, shape=())
        self.obstacle_pos = ti.Vector.field(3, dtype=float, shape=(4,))
        self.obstacle_vel = ti.Vector.field(3, dtype=float, shape=(4,))
        self.obstacle_radius = ti.field(dtype=float, shape=(4,))
        self.obstacle_type = ti.field(dtype=int, shape=(4,))          # 0=sphere, 1=box
        self.obstacle_rotation = ti.Vector.field(4, dtype=float, shape=(4,))  # quaternion (w,x,y,z)
        self.obstacle_size = ti.Vector.field(3, dtype=float, shape=(4,))      # box half-extents
        self.obstacle_count[None] = 0
        for o in range(4):
            self.obstacle_rotation[o] = [1.0, 0.0, 0.0, 0.0]
            self.obstacle_size[o] = [0.0, 0.0, 0.0]

    # ---- Scene Initialization ----

    @ti.kernel
    def init_dam_break(self):
        dx = self.dx
        sp = dx * 0.7
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        box_x = hi_x - lo_x
        box_y = hi_y - lo_y
        box_z = hi_z - lo_z

        npx = int(box_x / sp)
        npy = int(box_y / sp)
        npz = int(box_z / sp)
        max_particles = npx * npy * npz

        for i in range(max_particles):
            ix = i % npx
            iy = (i // npx) % npy
            iz = i // (npx * npy)
            self.pos[i] = [
                lo_x + (ix + 0.5) * sp,
                lo_y + (iy + 0.5) * sp,
                lo_z + (iz + 0.5) * sp,
            ]
            self.vel[i] = [0.0, 0.0, 0.0]

        for i in range(max_particles, self.num_particles):
            self.pos[i] = [-100.0, -100.0, -100.0]
            self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_dam_break_with_obstacle(self):
        """Dam break with a cylindrical obstacle in the middle."""
        dx = self.dx
        lo_x, hi_x = 0.05, 0.45
        lo_y, hi_y = 0.05, 0.85
        lo_z, hi_z = 0.05, 0.45
        box_x = hi_x - lo_x
        box_y = hi_y - lo_y
        box_z = hi_z - lo_z

        # Cylinder obstacle configuration
        obs_radius = 0.04  # Radius of the cylindrical obstacle (reduced)
        obs_height = 0.4   # Height of the cylinder (reduced from 0.6)
        obs_center_y = 0.5 # Center of cylinder in Y (moved up)

        npx = int(box_x / (dx * 0.7))
        npy = int(box_y / (dx * 0.7))
        npz = int(box_z / (dx * 0.7))
        max_particles = npx * npy * npz

        # Set obstacle (1 cylinder at center)
        self.obstacle_count[None] = 1
        self.obstacle_pos[0] = [0.25, obs_center_y, 0.25]
        self.obstacle_vel[0] = [0.0, 0.0, 0.0]
        self.obstacle_radius[0] = obs_radius

        sp = dx * 0.7
        for i in range(max_particles):
            ix = i % npx
            iy = (i // npx) % npy
            iz = i // (npx * npy)
            x = lo_x + (ix + 0.5) * sp
            y = lo_y + (iy + 0.5) * sp
            z = lo_z + (iz + 0.5) * sp

            # Check if particle is inside the cylinder obstacle
            dist_xy = (x - 0.25) ** 2 + (z - 0.25) ** 2
            in_cylinder = dist_xy < obs_radius ** 2 and abs(y - obs_center_y) < obs_height / 2

            if not in_cylinder:
                self.pos[i] = [x, y, z]
            else:
                # Push particles outside the obstacle
                if dist_xy < obs_radius ** 2:
                    # Push outward in XY plane
                    angle = ti.atan2(z - 0.25, x - 0.25)
                    x = 0.25 + (obs_radius + 0.01) * ti.cos(angle)
                    z = 0.25 + (obs_radius + 0.01) * ti.sin(angle)
                if abs(y - obs_center_y) < obs_height / 2:
                    # Push up or down
                    if y > obs_center_y:
                        y = obs_center_y + obs_height / 2 + 0.01
                    else:
                        y = obs_center_y - obs_height / 2 - 0.01

                self.pos[i] = [x, y, z]

            self.vel[i] = [0.0, 0.0, 0.0]

        # Mark any extra particles as inactive
        for i in range(max_particles, self.num_particles):
            self.pos[i] = [-100.0, -100.0, -100.0]
            self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_drop(self):
        dx = self.dx
        sp = dx * 0.7
        cx, cy, cz = 0.5, 0.75, 0.5
        radius = 0.15
        lo_x = cx - radius
        hi_x = cx + radius
        lo_y = cy - radius
        hi_y = cy + radius
        lo_z = cz - radius
        hi_z = cz + radius

        box_x = hi_x - lo_x
        box_y = hi_y - lo_y
        box_z = hi_z - lo_z
        npx = int(box_x / sp)
        npy = int(box_y / sp)
        npz = int(box_z / sp)
        max_particles = npx * npy * npz

        for i in range(self.num_particles):
            if i < max_particles:
                ix = i % npx
                iy = (i // npx) % npy
                iz = i // (npx * npy)
                x = lo_x + (ix + 0.5) * sp
                y = lo_y + (iy + 0.5) * sp
                z = lo_z + (iz + 0.5) * sp

                dx_dist = x - cx
                dy_dist = y - cy
                dz_dist = z - cz
                dist2 = dx_dist * dx_dist + dy_dist * dy_dist + dz_dist * dz_dist

                if dist2 <= radius * radius:
                    self.pos[i] = [x, y, z]
                else:
                    self.pos[i] = [-100.0, -100.0, -100.0]
            else:
                self.pos[i] = [-100.0, -100.0, -100.0]
            self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_double_dam(self):
        dx = self.dx
        sp = dx * 0.7
        lo_x1, hi_x1 = 0.05, 0.25
        lo_y1, hi_y1 = 0.05, 0.65
        lo_z1, hi_z1 = 0.05, 0.45
        npx1 = int((hi_x1 - lo_x1) / sp)
        npy1 = int((hi_y1 - lo_y1) / sp)
        npz1 = int((hi_z1 - lo_z1) / sp)
        count1 = npx1 * npy1 * npz1
        lo_x2, hi_x2 = 0.75, 0.95
        lo_y2, hi_y2 = 0.05, 0.65
        lo_z2, hi_z2 = 0.55, 0.95
        npx2 = int((hi_x2 - lo_x2) / sp)
        npy2 = int((hi_y2 - lo_y2) / sp)
        npz2 = int((hi_z2 - lo_z2) / sp)
        total = count1 + npx2 * npy2 * npz2

        for i in range(self.num_particles):
            if i < count1:
                ix = i % npx1
                iy = (i // npx1) % npy1
                iz = i // (npx1 * npy1)
                self.pos[i] = [
                    lo_x1 + (ix + 0.5) * sp,
                    lo_y1 + (iy + 0.5) * sp,
                    lo_z1 + (iz + 0.5) * sp,
                ]
            elif i < total:
                idx = i - count1
                ix = idx % npx2
                iy = (idx // npx2) % npy2
                iz = idx // (npx2 * npy2)
                self.pos[i] = [
                    lo_x2 + (ix + 0.5) * sp,
                    lo_y2 + (iy + 0.5) * sp,
                    lo_z2 + (iz + 0.5) * sp,
                ]
            else:
                self.pos[i] = [-100.0, -100.0, -100.0]
            self.vel[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def init_colors(self):
        for i in range(self.num_particles):
            self.color[i] = [0.2, 0.5, 1.0]

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
        dx = self.dx
        for o in ti.static(range(self._max_obstacles)):
            if o < self.obstacle_count[None]:
                obs_pos = self.obstacle_pos[o]
                if self.obstacle_type[o] == 0:  # sphere
                    obs_r = self.obstacle_radius[o]
                    if obs_r <= 0:
                        continue
                    r2 = (obs_r + dx) ** 2
                    i0 = ti.max(0, int((obs_pos[0] - obs_r - dx) / dx))
                    i1 = ti.min(self.nx - 1, int((obs_pos[0] + obs_r + dx) / dx) + 1)
                    j0 = ti.max(0, int((obs_pos[1] - obs_r - dx) / dx))
                    j1 = ti.min(self.ny - 1, int((obs_pos[1] + obs_r + dx) / dx) + 1)
                    k0 = ti.max(0, int((obs_pos[2] - obs_r - dx) / dx))
                    k1 = ti.min(self.nz - 1, int((obs_pos[2] + obs_r + dx) / dx) + 1)
                    for i, j, k in ti.ndrange((i0, i1 + 1), (j0, j1 + 1), (k0, k1 + 1)):
                        cx = (i + 0.5) * dx
                        cy = (j + 0.5) * dx
                        cz = (k + 0.5) * dx
                        dist2 = (cx - obs_pos[0])**2 + (cy - obs_pos[1])**2 + (cz - obs_pos[2])**2
                        if dist2 < r2:
                            self.cell_type[i, j, k] = 2
                else:  # box
                    sz = self.obstacle_size[o]
                    half_x, half_y, half_z = sz[0] + dx, sz[1] + dx, sz[2] + dx
                    qw = self.obstacle_rotation[o][0]
                    qx = self.obstacle_rotation[o][1]
                    qy = self.obstacle_rotation[o][2]
                    qz = self.obstacle_rotation[o][3]
                    cqw, cqx, cqy, cqz = qw, -qx, -qy, -qz
                    bound = ti.sqrt(half_x*half_x + half_y*half_y + half_z*half_z)
                    i0 = ti.max(0, int((obs_pos[0] - bound) / dx))
                    i1 = ti.min(self.nx - 1, int((obs_pos[0] + bound) / dx) + 1)
                    j0 = ti.max(0, int((obs_pos[1] - bound) / dx))
                    j1 = ti.min(self.ny - 1, int((obs_pos[1] + bound) / dx) + 1)
                    k0 = ti.max(0, int((obs_pos[2] - bound) / dx))
                    k1 = ti.min(self.nz - 1, int((obs_pos[2] + bound) / dx) + 1)
                    for i, j, k in ti.ndrange((i0, i1 + 1), (j0, j1 + 1), (k0, k1 + 1)):
                        cx = (i + 0.5) * dx
                        cy = (j + 0.5) * dx
                        cz = (k + 0.5) * dx
                        lx = cx - obs_pos[0]
                        ly = cy - obs_pos[1]
                        lz = cz - obs_pos[2]
                        tw = -cqx*lx - cqy*ly - cqz*lz
                        tx = cqw*lx + cqy*lz - cqz*ly
                        ty = cqw*ly - cqx*lz + cqz*lx
                        tz = cqw*lz + cqx*ly - cqy*lx
                        rx = tw*qx + tx*qw + ty*qz - tz*qy
                        ry = tw*qy - tx*qz + ty*qw + tz*qx
                        rz = tw*qz + tx*qy - ty*qx + tz*qw
                        if ti.abs(rx) < half_x and ti.abs(ry) < half_y and ti.abs(rz) < half_z:
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
    def relabel_and_density(self):
        """Combined: relabel cells + update particle density + compute target density."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 2:
                self.cell_type[i, j, k] = 1  # air
            self.particle_density[i, j, k] = 0.0
        for p in range(self.num_particles):
            if self.pos[p][0] < 0:
                continue
            ci = int(self.pos[p][0] / self.dx)
            cj = int(self.pos[p][1] / self.dx)
            ck = int(self.pos[p][2] / self.dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                self.cell_type[ci, cj, ck] = 0  # fluid
                self.particle_density[ci, cj, ck] += 1.0

    @ti.kernel
    def _compute_target_density(self):
        """Compute global average density: total particles in fluid cells / fluid cell count."""
        total = 0.0
        cells = 0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 0:
                total += self.particle_density[i, j, k]
                cells += 1
        self._target_density[None] = total / max(cells, 1)

    # ---- Simulation Kernels ----

    @ti.kernel
    def integrate_and_collide(self, dt: float, gravity: float):
        """Single kernel: integrate particles, then clamp domain + obstacle collision."""
        eps = 1e-6
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            # Integrate
            self.vel[i][1] += gravity * dt
            self.pos[i] += self.vel[i] * dt
            # Domain boundary
            for d in ti.static(range(3)):
                if self.pos[i][d] < eps:
                    self.pos[i][d] = eps
                    if self.vel[i][d] < 0:
                        self.vel[i][d] = 0.0
                elif self.pos[i][d] > 1.0 - eps:
                    self.pos[i][d] = 1.0 - eps
                    if self.vel[i][d] > 0:
                        self.vel[i][d] = 0.0
            # Obstacle collisions
            for o in ti.static(range(self._max_obstacles)):
                if o < self.obstacle_count[None]:
                    if self.obstacle_type[o] == 0:  # sphere
                        obs_r = self.obstacle_radius[o]
                        if obs_r <= 0:
                            continue
                        obs_pos = self.obstacle_pos[o]
                        obs_r_col = obs_r + 0.01
                        diff = self.pos[i] - obs_pos
                        dist = diff.norm()
                        if dist < obs_r_col and dist > 1e-8:
                            n = diff / dist
                            self.pos[i] = obs_pos + n * obs_r_col
                            obs_vel = self.obstacle_vel[o]
                            rel_vel = self.vel[i] - obs_vel
                            vn = rel_vel.dot(n)
                            if vn < 0:
                                self.vel[i] -= n * vn * 1.5
                                self.vel[i] += obs_vel
                    else:  # box
                        sz = self.obstacle_size[o]
                        obs_pos = self.obstacle_pos[o]
                        qw = self.obstacle_rotation[o][0]
                        qx = self.obstacle_rotation[o][1]
                        qy = self.obstacle_rotation[o][2]
                        qz = self.obstacle_rotation[o][3]
                        cqw, cqx, cqy, cqz = qw, -qx, -qy, -qz
                        lx = self.pos[i][0] - obs_pos[0]
                        ly = self.pos[i][1] - obs_pos[1]
                        lz = self.pos[i][2] - obs_pos[2]
                        tw = -cqx*lx - cqy*ly - cqz*lz
                        tx = cqw*lx + cqy*lz - cqz*ly
                        ty = cqw*ly - cqx*lz + cqz*lx
                        tz = cqw*lz + cqx*ly - cqy*lx
                        rx = tw*qx + tx*qw + ty*qz - tz*qy
                        ry = tw*qy - tx*qz + ty*qw + tz*qx
                        rz = tw*qz + tx*qy - ty*qx + tz*qw
                        half_x, half_y, half_z = sz[0] + 0.01, sz[1] + 0.01, sz[2] + 0.01
                        push_x = ti.max(0.0, ti.abs(rx) - half_x)
                        push_y = ti.max(0.0, ti.abs(ry) - half_y)
                        push_z = ti.max(0.0, ti.abs(rz) - half_z)
                        if push_x > 0 or push_y > 0 or push_z > 0:
                            sign_x = -1.0 if rx < 0 else 1.0
                            sign_y = -1.0 if ry < 0 else 1.0
                            sign_z = -1.0 if rz < 0 else 1.0
                            lpx, lpy, lpz = sign_x * push_x, sign_y * push_y, sign_z * push_z
                            ptw = -qx*lpx - qy*lpy - qz*lpz
                            ptx = qw*lpx + qy*lpz - qz*lpy
                            pty = qw*lpy - qx*lpz + qz*lpx
                            ptz = qw*lpz + qx*lpy - qy*lpx
                            wpx = ptw*cqx + ptx*cqw + pty*cqz - ptz*cqy
                            wpy = ptw*cqy - ptx*cqz + pty*cqw + ptz*cqx
                            wpz = ptw*cqz + ptx*cqy - pty*cqx + ptz*cqw
                            self.pos[i][0] += wpx
                            self.pos[i][1] += wpy
                            self.pos[i][2] += wpz
                            obs_vel = self.obstacle_vel[o]
                            rel_vel = self.vel[i] - obs_vel
                            vn = rel_vel[0]*wpx + rel_vel[1]*wpy + rel_vel[2]*wpz
                            push_norm = ti.sqrt(wpx*wpx + wpy*wpy + wpz*wpz)
                            if push_norm > 1e-8:
                                if vn < 0:
                                    nx = wpx / push_norm
                                    ny = wpy / push_norm
                                    nz = wpz / push_norm
                                    self.vel[i][0] -= nx * vn * 1.5
                                    self.vel[i][1] -= ny * vn * 1.5
                                    self.vel[i][2] -= nz * vn * 1.5
                                    self.vel[i] += obs_vel

    def handle_particle_collisions(self):
        """Compatibility: use integrate_and_collide with zero dt."""
        self.integrate_and_collide(0.0, 0.0)

    def push_particles_apart(self, num_iters: int):
        for _ in range(num_iters):
            self.grid_particle_count.fill(0)
            self._build_neighbor_grid()
            self._separate_pass()

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
        min_dist = dx * 0.3
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
                    for s in range(cnt):
                        q = self.grid_particle_ids[ni, nj, nk, s]
                        if q != p:
                            diff = self.pos[p] - self.pos[q]
                            d2 = diff.dot(diff)
                            if d2 < min_dist2 and d2 > 1e-12:
                                d = ti.sqrt(d2)
                                n_dir = diff / d
                                self.pos[p] += n_dir * (min_dist - d) * 0.3

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

    def store_initial_density(self):
        self._compute_target_density()

    @ti.kernel
    def apply_horizontal_impulse(self, impulse_x: float, impulse_z: float):
        """Apply a horizontal velocity impulse to all active particles."""
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                continue
            self.vel[i][0] += impulse_x
            self.vel[i][2] += impulse_z

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
                    - self._target_density[None]
                )
                if density_diff > 0.0:
                    div -= density_diff * 0.2
                else:
                    div -= density_diff * 0.08

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

    # ---- CG Pressure Projection ----

    def solve_incompressibility_cg(
        self,
        num_iters: int,
        dt: float,
        compensate_drift: bool,
    ):
        """Conjugate Gradient pressure solve. Modifies grid velocities directly."""
        self.pressure.fill(0)
        self._cg_init_residual(compensate_drift)
        self._cg_p.copy_from(self._cg_r)
        self._cg_rdotr[0] = 0.0
        self._cg_compute_rdotr()
        rr = self._cg_rdotr[0]
        if rr < 1e-14:
            return
        for _ in range(num_iters):
            self._cg_Ap.fill(0)
            self._cg_compute_Ap()
            self._cg_pAp[0] = 0.0
            self._cg_compute_pAp()
            pAp = self._cg_pAp[0]
            if abs(pAp) < 1e-14:
                break
            alpha = rr / pAp
            self._cg_update(alpha)
            old_rr = rr
            self._cg_rdotr[0] = 0.0
            self._cg_compute_rdotr()
            rr = self._cg_rdotr[0]
            if rr < 1e-14:
                break
            beta = rr / old_rr
            self._cg_update_p(beta)

        # Apply pressure gradient to velocity
        self._cg_apply_pressure_gradient()

    @ti.kernel
    def _cg_init_residual(self, compensate_drift: bool):
        """Initialize residual r = b - Ax, with x=0 so r = b (divergence)."""
        self._cg_rdotr[0] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 0:
                self._cg_r[i, j, k] = 0.0
                continue
            div = (
                self.grid_u[i + 1, j, k]
                - self.grid_u[i, j, k]
                + self.grid_v[i, j + 1, k]
                - self.grid_v[i, j, k]
                + self.grid_w[i, j, k + 1]
                - self.grid_w[i, j, k]
            )
            if compensate_drift:
                density_diff = (
                    self.particle_density[i, j, k]
                    - self._target_density[None]
                )
                if density_diff > 0.0:
                    div -= density_diff * 0.2
                else:
                    div -= density_diff * 0.08
            self._cg_r[i, j, k] = div

    @ti.kernel
    def _cg_compute_rdotr(self):
        s = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            s += self._cg_r[i, j, k] * self._cg_r[i, j, k]
        self._cg_rdotr[0] = s

    @ti.kernel
    def _cg_compute_Ap(self):
        """Compute Ap = L * p where L is the discrete Laplacian."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 0:
                continue
            s = 0.0
            if i > 0 and self.cell_type[i - 1, j, k] != 2:
                s -= self._cg_p[i - 1, j, k]
            else:
                s -= 0.0
            if i < self.nx - 1 and self.cell_type[i + 1, j, k] != 2:
                s -= self._cg_p[i + 1, j, k]
            if j > 0 and self.cell_type[i, j - 1, k] != 2:
                s -= self._cg_p[i, j - 1, k]
            if j < self.ny - 1 and self.cell_type[i, j + 1, k] != 2:
                s -= self._cg_p[i, j + 1, k]
            if k > 0 and self.cell_type[i, j, k - 1] != 2:
                s -= self._cg_p[i, j, k - 1]
            if k < self.nz - 1 and self.cell_type[i, j, k + 1] != 2:
                s -= self._cg_p[i, j, k + 1]
            n_neighbors = 0.0
            if i > 0 and self.cell_type[i - 1, j, k] != 2:
                n_neighbors += 1.0
            if i < self.nx - 1 and self.cell_type[i + 1, j, k] != 2:
                n_neighbors += 1.0
            if j > 0 and self.cell_type[i, j - 1, k] != 2:
                n_neighbors += 1.0
            if j < self.ny - 1 and self.cell_type[i, j + 1, k] != 2:
                n_neighbors += 1.0
            if k > 0 and self.cell_type[i, j, k - 1] != 2:
                n_neighbors += 1.0
            if k < self.nz - 1 and self.cell_type[i, j, k + 1] != 2:
                n_neighbors += 1.0
            self._cg_Ap[i, j, k] = n_neighbors * self._cg_p[i, j, k] + s

    @ti.kernel
    def _cg_compute_pAp(self):
        s = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            s += self._cg_p[i, j, k] * self._cg_Ap[i, j, k]
        self._cg_pAp[0] = s

    @ti.kernel
    def _cg_update(self, alpha: float):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.pressure[i, j, k] += alpha * self._cg_p[i, j, k]
            self._cg_r[i, j, k] -= alpha * self._cg_Ap[i, j, k]

    @ti.kernel
    def _cg_update_p(self, beta: float):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self._cg_p[i, j, k] = self._cg_r[i, j, k] + beta * self._cg_p[i, j, k]

    @ti.kernel
    def _cg_apply_pressure_gradient(self):
        """Apply pressure gradient to staggered grid velocities."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] != 0:
                continue
            if i > 0 and self.cell_type[i - 1, j, k] != 2:
                self.grid_u[i, j, k] += self.pressure[i, j, k] - self.pressure[i - 1, j, k]
            if i < self.nx - 1 and self.cell_type[i + 1, j, k] != 2:
                self.grid_u[i + 1, j, k] -= self.pressure[i + 1, j, k] - self.pressure[i, j, k]
            if j > 0 and self.cell_type[i, j - 1, k] != 2:
                self.grid_v[i, j, k] += self.pressure[i, j, k] - self.pressure[i, j - 1, k]
            if j < self.ny - 1 and self.cell_type[i, j + 1, k] != 2:
                self.grid_v[i, j + 1, k] -= self.pressure[i, j + 1, k] - self.pressure[i, j, k]
            if k > 0 and self.cell_type[i, j, k - 1] != 2:
                self.grid_w[i, j, k] += self.pressure[i, j, k] - self.pressure[i, j, k - 1]
            if k < self.nz - 1 and self.cell_type[i, j, k + 1] != 2:
                self.grid_w[i, j, k + 1] -= self.pressure[i, j, k + 1] - self.pressure[i, j, k]

    # ---- Grid Utilities ----

    def clear_grid(self):
        self.grid_u.fill(0)
        self.grid_u_weight.fill(0)
        self.grid_v.fill(0)
        self.grid_v_weight.fill(0)
        self.grid_w.fill(0)
        self.grid_w_weight.fill(0)

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
                self.color[i] = [0.1 + 0.9 * t, 0.2 + 0.7 * t, 0.9 - 0.7 * t]
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

    # ---- Debug ----

    @ti.kernel
    def debug_save_positions(self):
        for i in range(self.num_particles):
            self._prev_pos[i] = self.pos[i]

    @ti.kernel
    def debug_count_stuck(self) -> int:
        """Count particles with pos < 0 (permanently stuck)."""
        count = 0
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                count += 1
        return count

    @ti.kernel
    def debug_color_stuck_red(self):
        """Color stuck (pos < 0) particles bright red for visibility."""
        for i in range(self.num_particles):
            if self.pos[i][0] < 0:
                self.color[i] = [1.0, 0.0, 0.0]

    @ti.kernel
    def clamp_all_positions(self):
        """Clamp active particles into the domain. Skip truly inactive ones (x << 0)."""
        eps = 1e-6
        for i in range(self.num_particles):
            if self.pos[i][0] < -1.0:
                continue
            for d in ti.static(range(3)):
                if self.pos[i][d] < eps:
                    self.pos[i][d] = eps
                elif self.pos[i][d] > 1.0 - eps:
                    self.pos[i][d] = 1.0 - eps

    @ti.kernel
    def compute_fluid_volume(self):
        """Count fluid cells as a proxy for volume. Store ratio to init in _fluid_vol_ratio."""
        count = 0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.cell_type[i, j, k] == 0:
                count += 1
        self._fluid_cell_count[0] = count
        init = self._init_fluid_cells[0]
        ratio = 1.0 if init > 0 else 1.0
        if init > 0:
            ratio = float(count) / float(init)
        self._fluid_vol_ratio[0] = ratio

    def save_init_volume(self):
        """Call once after initialization to save initial fluid cell count."""
        self.compute_fluid_volume()
        self._init_fluid_cells[0] = self._fluid_cell_count[0]

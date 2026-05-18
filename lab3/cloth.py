import numpy as np
import taichi as ti

from lab3.collision import CollisionWorld
from lab3.constants import ConstraintMode, FEMConfig
from lab3.models import stvk_first_piola_tri


def _build_cloth_mesh(nx: int, ny: int, sx: float, sy: float):
    points = []
    uv = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u = sx * i / nx
            v = sy * j / ny
            points.append([u, 1.5, v])
            uv.append([u, v])

    def vid(i: int, j: int):
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    edge_set = set()
    for t in tris:
        a, b, c = t
        for i, j in [(a, b), (b, c), (c, a)]:
            edge_set.add((i, j) if i < j else (j, i))
    edges = np.array(sorted(edge_set), dtype=np.int32)

    bend_edges = []
    # 2-hop grid edges for simple bending stiffness
    for j in range(ny + 1):
        for i in range(nx + 1):
            if i + 2 <= nx:
                bend_edges.append([vid(i, j), vid(i + 2, j)])
            if j + 2 <= ny:
                bend_edges.append([vid(i, j), vid(i, j + 2)])

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(uv, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
        edges,
        np.asarray(bend_edges, dtype=np.int32),
    )


@ti.data_oriented
class ClothSystem:
    def __init__(self, config: FEMConfig, nx: int = 24, ny: int = 24, sx: float = 2.0, sy: float = 2.0):
        self.config = config
        points, uv, tris, edges, bend_edges = _build_cloth_mesh(nx, ny, sx, sy)

        self.num_vertices = points.shape[0]
        self.num_tris = tris.shape[0]
        self.num_edges = edges.shape[0]
        self.num_bend_edges = bend_edges.shape[0]

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.fixed = ti.field(dtype=ti.i32, shape=self.num_vertices)

        self.tris = ti.Vector.field(3, dtype=ti.i32, shape=self.num_tris)
        self.dm_inv = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_tris)
        self.rest_area = ti.field(dtype=ti.f32, shape=self.num_tris)
        self.edge_indices = ti.field(dtype=ti.i32, shape=self.num_edges * 2)
        self.bend_edges = ti.Vector.field(2, dtype=ti.i32, shape=self.num_bend_edges)
        self.bend_rest_len = ti.field(dtype=ti.f32, shape=self.num_bend_edges)
        self.line_points = ti.Vector.field(3, dtype=ti.f32, shape=self.num_edges * 2)

        self.drag_vertex_idx = ti.field(dtype=ti.i32, shape=())
        self.drag_force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.drag_vertex_idx[None] = -1
        self.drag_force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.drag_stiffness = 1500.0
        self.drag_damping = 20.0
        self.pick_radius = 0.12
        self._drag_t = 0.0
        self.collision_world: CollisionWorld | None = None

        self._init_from_numpy(points, uv, tris, edges, bend_edges, nx, ny)
        self._rest_positions_np = self.x.to_numpy()
        self._nx = nx
        self._ny = ny

    def _init_from_numpy(self, points, uv, tris, edges, bend_edges, nx, ny):
        self.x.from_numpy(points)
        self.v.fill(0.0)
        self.f.fill(0.0)
        self.tris.from_numpy(tris)
        self.fixed.fill(0)

        dm_inv_np = np.zeros((self.num_tris, 2, 2), dtype=np.float32)
        area_np = np.zeros(self.num_tris, dtype=np.float32)
        masses = np.zeros(self.num_vertices, dtype=np.float32)

        for e, tri in enumerate(tris):
            u0, v0 = uv[tri[0]]
            u1, v1 = uv[tri[1]]
            u2, v2 = uv[tri[2]]
            Dm = np.array([[u1 - u0, u2 - u0], [v1 - v0, v2 - v0]], dtype=np.float32)
            det = np.linalg.det(Dm)
            if abs(det) < 1.0e-10:
                raise ValueError(f"Degenerate triangle {e}")
            dm_inv_np[e] = np.linalg.inv(Dm).astype(np.float32)
            area = 0.5 * abs(det)
            area_np[e] = area
            lump = self.config.density * area / 3.0
            masses[tri[0]] += lump
            masses[tri[1]] += lump
            masses[tri[2]] += lump

        self.dm_inv.from_numpy(dm_inv_np)
        self.rest_area.from_numpy(area_np)
        self.mass.from_numpy(masses)
        inv = np.zeros_like(masses)
        mask = masses > 1.0e-12
        inv[mask] = 1.0 / masses[mask]
        self.inv_mass.from_numpy(inv)
        self.edge_indices.from_numpy(edges.reshape(-1))
        self.bend_edges.from_numpy(bend_edges)
        if self.num_bend_edges > 0:
            p0 = points[bend_edges[:, 0]]
            p1 = points[bend_edges[:, 1]]
            rest = np.linalg.norm(p1 - p0, axis=1).astype(np.float32)
            self.bend_rest_len.from_numpy(rest)

        self._apply_constraint_mode(points, nx, ny, self.config.constraint_mode)

    def _apply_constraint_mode(self, points: np.ndarray, nx: int, ny: int, mode: ConstraintMode):
        x = points[:, 0]
        y = points[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        tol = 1.0e-6
        pinned = np.zeros(self.num_vertices, dtype=np.int32)
        if mode == ConstraintMode.TOP:
            pinned[np.isclose(y, y_max, atol=tol)] = 1
        elif mode == ConstraintMode.SIDE_X_MIN:
            pinned[np.isclose(x, x_min, atol=tol)] = 1
        elif mode == ConstraintMode.SIDE_X_BOTH:
            pinned[np.isclose(x, x_min, atol=tol) | np.isclose(x, x_max, atol=tol)] = 1
        elif mode == ConstraintMode.TOP_BOTTOM:
            pinned[np.isclose(y, y_min, atol=tol) | np.isclose(y, y_max, atol=tol)] = 1
        else:
            for i in range(nx + 1):
                pinned[i] = 1
        self.fixed.from_numpy(pinned)

    @ti.kernel
    def clear_forces(self):
        for i in self.f:
            self.f[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def add_gravity(self, gx: ti.f32, gy: ti.f32, gz: ti.f32):
        g = ti.Vector([gx, gy, gz])
        for i in self.f:
            if self.fixed[i] == 0:
                self.f[i] += self.mass[i] * g

    @ti.kernel
    def accumulate_internal_forces(self, material_type: ti.i32, mu: ti.f32, lmbda: ti.f32):
        for e in range(self.num_tris):
            tri = self.tris[e]
            i0, i1, i2 = tri[0], tri[1], tri[2]
            x0, x1, x2 = self.x[i0], self.x[i1], self.x[i2]
            Ds = ti.Matrix.cols([x1 - x0, x2 - x0])  # 3x2
            F = Ds @ self.dm_inv[e]  # 3x2

            # B2 baseline: StVK on 3x2 F.
            P = stvk_first_piola_tri(F, mu, lmbda)

            H = -self.rest_area[e] * P @ self.dm_inv[e].transpose()  # 3x2
            f1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            f2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            f0 = -(f1 + f2)

            for c in ti.static(range(3)):
                ti.atomic_add(self.f[i0][c], f0[c])
                ti.atomic_add(self.f[i1][c], f1[c])
                ti.atomic_add(self.f[i2][c], f2[c])

    @ti.kernel
    def add_drag_force(self):
        idx = self.drag_vertex_idx[None]
        if idx >= 0 and self.fixed[idx] == 0:
            self.f[idx] += self.drag_force[None]

    @ti.kernel
    def add_bending_forces(self, bend_k: ti.f32, bend_damping: ti.f32):
        for e in range(self.num_bend_edges):
            i0 = self.bend_edges[e][0]
            i1 = self.bend_edges[e][1]
            if self.fixed[i0] == 1 and self.fixed[i1] == 1:
                continue
            x0 = self.x[i0]
            x1 = self.x[i1]
            d = x1 - x0
            l = d.norm() + 1.0e-8
            n = d / l
            stretch = l - self.bend_rest_len[e]
            rel_v = self.v[i1] - self.v[i0]
            vn = rel_v.dot(n)
            fmag = bend_k * stretch + bend_damping * vn
            fvec = fmag * n
            if self.fixed[i0] == 0:
                for c in ti.static(range(3)):
                    ti.atomic_add(self.f[i0][c], fvec[c])
            if self.fixed[i1] == 0:
                for c in ti.static(range(3)):
                    ti.atomic_add(self.f[i1][c], -fvec[c])

    @ti.kernel
    def integrate_explicit(self, dt: ti.f32, damping: ti.f32):
        for i in self.x:
            if self.fixed[i] == 1:
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])
                continue
            self.v[i] = damping * (self.v[i] + dt * self.inv_mass[i] * self.f[i])
            self.x[i] += dt * self.v[i]

    @ti.kernel
    def apply_ground_plane(self, y_ground: ti.f32, restitution: ti.f32):
        for i in self.x:
            if self.x[i].y < y_ground:
                self.x[i].y = y_ground
                if self.v[i].y < 0.0:
                    self.v[i].y = -restitution * self.v[i].y

    @ti.kernel
    def build_line_points(self):
        for e in range(self.num_edges):
            i0 = self.edge_indices[2 * e]
            i1 = self.edge_indices[2 * e + 1]
            self.line_points[2 * e] = self.x[i0]
            self.line_points[2 * e + 1] = self.x[i1]

    def begin_drag(self, ray_origin, ray_dir):
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_dir, dtype=np.float32)
        n = np.linalg.norm(direction)
        if n < 1.0e-8:
            self.drag_vertex_idx[None] = -1
            self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return
        direction /= n

        x_np = self.x.to_numpy()
        hit_idx = -1
        hit_t = np.inf
        r2 = self.pick_radius * self.pick_radius
        for i in range(self.num_vertices):
            oc = origin - x_np[i]
            b = float(np.dot(direction, oc))
            c_term = float(np.dot(oc, oc) - r2)
            disc = b * b - c_term
            if disc < 0.0:
                continue
            s = np.sqrt(disc)
            t0 = -b - s
            t1 = -b + s
            t = t0 if t0 > 0.0 else t1
            if t > 0.0 and t < hit_t:
                hit_t = t
                hit_idx = i
        self.drag_vertex_idx[None] = hit_idx
        self._drag_t = float(hit_t if hit_idx >= 0 else 0.0)
        self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def drag_to(self, ray_origin, ray_dir):
        idx = int(self.drag_vertex_idx[None])
        if idx < 0:
            return
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_dir, dtype=np.float32)
        n = np.linalg.norm(direction)
        if n < 1.0e-8:
            return
        direction /= n
        target = origin + self._drag_t * direction
        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()
        disp = target - x_np[idx]
        force = self.drag_stiffness * disp - self.drag_damping * v_np[idx]
        self.drag_force[None] = force.astype(np.float32)

    def end_drag(self):
        self.drag_vertex_idx[None] = -1
        self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_constraint_mode(self, mode: ConstraintMode) -> None:
        self.config.constraint_mode = mode
        self._apply_constraint_mode(self._rest_positions_np, self._nx, self._ny, mode)
        self.end_drag()

    def reset_state(self) -> None:
        self.x.from_numpy(self._rest_positions_np)
        self.v.fill(0.0)
        self.f.fill(0.0)
        self.end_drag()

    def set_collision_world(self, world: CollisionWorld | None) -> None:
        self.collision_world = world

    def add_collision_forces(self) -> None:
        if not self.config.enable_collision or self.collision_world is None:
            return
        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()
        f_np = self.f.to_numpy()
        fixed_np = self.fixed.to_numpy()
        k = float(self.config.collision_k)
        c = float(self.config.collision_c)
        pr = float(self.config.collision_particle_radius)
        iters = max(1, int(self.config.collision_iters))

        for _ in range(iters):
            for i in range(self.num_vertices):
                if fixed_np[i] == 1:
                    continue
                xi = x_np[i]
                vi = v_np[i]

                for col in self.collision_world.planes:
                    if not col.enabled:
                        continue
                    hit, n, depth = col.query(xi, pr)
                    if hit:
                        vn = float(np.dot(vi, n))
                        damp = -c * min(vn, 0.0)
                        f_np[i] += (k * depth + damp) * n

                for col in self.collision_world.spheres:
                    if not col.enabled:
                        continue
                    hit, n, depth = col.query(xi, pr)
                    if hit:
                        vn = float(np.dot(vi, n))
                        damp = -c * min(vn, 0.0)
                        f_np[i] += (k * depth + damp) * n

                for col in self.collision_world.aabbs:
                    if not col.enabled:
                        continue
                    hit, n, depth = col.query(xi, pr)
                    if hit:
                        vn = float(np.dot(vi, n))
                        damp = -c * min(vn, 0.0)
                        f_np[i] += (k * depth + damp) * n

        self.f.from_numpy(f_np.astype(np.float32))

import numpy as np
import taichi as ti

from lab3.collision import CollisionWorld
from lab3.constants import ConstraintMode, FEMConfig
from lab3.mesh import build_box_tet_mesh, extract_unique_edges
from lab3.models import corotated_first_piola, neo_hookean_first_piola, stvk_first_piola


@ti.data_oriented
class FEMSystem:
    def __init__(self, config: FEMConfig):
        self.config = config
        points, tets = build_box_tet_mesh(config.wx, config.wy, config.wz, config.cell_size)
        edges = extract_unique_edges(tets)

        self.num_vertices = points.shape[0]
        self.num_tets = tets.shape[0]
        self.num_edges = edges.shape[0]

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.f = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.fixed = ti.field(dtype=ti.i32, shape=self.num_vertices)

        self.tets = ti.Vector.field(4, dtype=ti.i32, shape=self.num_tets)
        self.dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_tets)
        self.rest_volume = ti.field(dtype=ti.f32, shape=self.num_tets)
        self.edge_indices = ti.field(dtype=ti.i32, shape=self.num_edges * 2)
        self.line_points = ti.Vector.field(3, dtype=ti.f32, shape=self.num_edges * 2)
        self.drag_vertex_idx = ti.field(dtype=ti.i32, shape=())
        self.drag_force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.drag_vertex_idx[None] = -1
        self.drag_force[None] = ti.Vector([0.0, 0.0, 0.0])

        self.drag_stiffness = 800.0
        self.drag_damping = 15.0
        self.pick_radius = 0.18
        self._drag_t = 0.0
        self.collision_world: CollisionWorld | None = None

        # Boundary vibration state
        self._sim_time = 0.0
        self._boundary_rest_y = ti.field(dtype=ti.f32, shape=self.num_vertices)

        self._init_from_numpy(points, tets, edges)
        self._rest_positions_np = self.x.to_numpy()

    def _init_from_numpy(self, points: np.ndarray, tets: np.ndarray, edges: np.ndarray) -> None:
        self.x.from_numpy(points)
        self.v.fill(0.0)
        self.f.fill(0.0)
        self.tets.from_numpy(tets)
        self.fixed.fill(0)

        dm_inv_np = np.zeros((self.num_tets, 3, 3), dtype=np.float32)
        rest_vol_np = np.zeros(self.num_tets, dtype=np.float32)
        masses = np.zeros(self.num_vertices, dtype=np.float32)

        for e, tet in enumerate(tets):
            X0 = points[tet[0]]
            X1 = points[tet[1]]
            X2 = points[tet[2]]
            X3 = points[tet[3]]
            Dm = np.column_stack((X1 - X0, X2 - X0, X3 - X0)).astype(np.float32)

            det_dm = np.linalg.det(Dm)
            if abs(det_dm) < 1.0e-10:
                raise ValueError(f"Degenerate tetrahedron detected at index {e}")

            dm_inv_np[e] = np.linalg.inv(Dm).astype(np.float32)
            vol = abs(det_dm) / 6.0
            rest_vol_np[e] = vol
            lump = self.config.density * vol / 4.0
            for vid in tet:
                masses[vid] += lump

        self.dm_inv.from_numpy(dm_inv_np)
        self.rest_volume.from_numpy(rest_vol_np)
        self.mass.from_numpy(masses)

        inv_mass = np.zeros_like(masses)
        nonzero = masses > 1.0e-12
        inv_mass[nonzero] = 1.0 / masses[nonzero]
        self.inv_mass.from_numpy(inv_mass)

        flat_edges = edges.reshape(-1)
        self.edge_indices.from_numpy(flat_edges)

        self._apply_constraint_mode(points, self.config.constraint_mode)

        # Store rest Y positions for fixed vertices (for vibration)
        boundary_rest_y_np = points[:, 1].copy()
        self._boundary_rest_y.from_numpy(boundary_rest_y_np)

    def _apply_constraint_mode(self, points: np.ndarray, mode: ConstraintMode) -> None:
        x = points[:, 0]
        y = points[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        tol = 1.0e-6
        pinned = np.zeros(points.shape[0], dtype=np.int32)
        if mode == ConstraintMode.TOP:
            pinned[np.isclose(y, y_max, atol=tol)] = 1
        elif mode == ConstraintMode.SIDE_X_MIN:
            pinned[np.isclose(x, x_min, atol=tol)] = 1
        elif mode == ConstraintMode.SIDE_X_BOTH:
            pinned[np.isclose(x, x_min, atol=tol) | np.isclose(x, x_max, atol=tol)] = 1
        elif mode == ConstraintMode.TOP_BOTTOM:
            pinned[np.isclose(y, y_min, atol=tol) | np.isclose(y, y_max, atol=tol)] = 1
        else:
            pinned[np.isclose(y, y_max, atol=tol)] = 1
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
        for e in range(self.num_tets):
            tet = self.tets[e]
            i0, i1, i2, i3 = tet[0], tet[1], tet[2], tet[3]
            x0, x1, x2, x3 = self.x[i0], self.x[i1], self.x[i2], self.x[i3]

            Ds = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
            F = Ds @ self.dm_inv[e]

            P = ti.Matrix.zero(ti.f32, 3, 3)
            if material_type == 0:
                P = stvk_first_piola(F, mu, lmbda)
            elif material_type == 1:
                P = neo_hookean_first_piola(F, mu, lmbda)
            else:
                P = corotated_first_piola(F, mu, lmbda)

            H = -self.rest_volume[e] * P @ self.dm_inv[e].transpose()
            f1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            f2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            f3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])
            f0 = -(f1 + f2 + f3)

            for c in ti.static(range(3)):
                ti.atomic_add(self.f[i0][c], f0[c])
                ti.atomic_add(self.f[i1][c], f1[c])
                ti.atomic_add(self.f[i2][c], f2[c])
                ti.atomic_add(self.f[i3][c], f3[c])

    @ti.kernel
    def integrate_explicit(self, dt: ti.f32, damping: ti.f32):
        for i in self.x:
            if self.fixed[i] == 1:
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])
                continue
            self.v[i] = damping * (self.v[i] + dt * self.inv_mass[i] * self.f[i])
            self.x[i] += dt * self.v[i]

    @ti.kernel
    def apply_boundary_vibration(self, t: ti.f32, amplitude: ti.f32, frequency: ti.f32):
        for i in self.x:
            if self.fixed[i] == 1:
                offset = amplitude * ti.sin(frequency * t)
                self.x[i].y = self._boundary_rest_y[i] + offset
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_ground_plane(self, y_ground: ti.f32, restitution: ti.f32):
        for i in self.x:
            if self.x[i].y < y_ground:
                self.x[i].y = y_ground
                if self.v[i].y < 0.0:
                    self.v[i].y = -restitution * self.v[i].y

    @ti.kernel
    def add_drag_force(self):
        idx = self.drag_vertex_idx[None]
        if idx >= 0 and self.fixed[idx] == 0:
            self.f[idx] += self.drag_force[None]

    @ti.kernel
    def build_line_points(self):
        for e in range(self.num_edges):
            i0 = self.edge_indices[2 * e]
            i1 = self.edge_indices[2 * e + 1]
            self.line_points[2 * e] = self.x[i0]
            self.line_points[2 * e + 1] = self.x[i1]

    # Interaction placeholder interfaces (to be implemented later)
    def begin_drag(self, ray_origin, ray_dir) -> None:
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_dir, dtype=np.float32)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1.0e-8:
            self.drag_vertex_idx[None] = -1
            self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return
        direction = direction / direction_norm

        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()

        hit_idx = -1
        hit_t = np.inf
        r2 = self.pick_radius * self.pick_radius
        for i in range(self.num_vertices):
            c = x_np[i]
            oc = origin - c
            b = float(np.dot(direction, oc))
            c_term = float(np.dot(oc, oc) - r2)
            disc = b * b - c_term
            if disc < 0.0:
                continue
            sqrt_disc = np.sqrt(disc)
            t0 = -b - sqrt_disc
            t1 = -b + sqrt_disc
            t = t0 if t0 > 0.0 else t1
            if t <= 0.0:
                continue
            if t < hit_t:
                hit_t = t
                hit_idx = i

        self.drag_vertex_idx[None] = hit_idx
        self._drag_t = float(hit_t if hit_idx >= 0 else 0.0)
        self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if hit_idx >= 0:
            _ = v_np  # keep local for future interaction extensions

    def begin_drag_vertex(self, vertex_idx: int, ray_origin, ray_dir) -> None:
        if vertex_idx < 0 or vertex_idx >= self.num_vertices:
            self.drag_vertex_idx[None] = -1
            self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_dir, dtype=np.float32)
        n = np.linalg.norm(direction)
        if n < 1.0e-8:
            self.drag_vertex_idx[None] = -1
            self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return
        direction /= n
        x_np = self.x.to_numpy()
        t = float(np.dot(x_np[vertex_idx] - origin, direction))
        if t < 0.0:
            t = float(np.linalg.norm(x_np[vertex_idx] - origin))
        self.drag_vertex_idx[None] = int(vertex_idx)
        self._drag_t = t
        self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def drag_to(self, ray_origin, ray_dir) -> None:
        idx = int(self.drag_vertex_idx[None])
        if idx < 0:
            return
        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_dir, dtype=np.float32)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1.0e-8:
            return
        direction = direction / direction_norm

        target = origin + self._drag_t * direction
        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()
        displacement = target - x_np[idx]
        force = self.drag_stiffness * displacement - self.drag_damping * v_np[idx]
        self.drag_force[None] = force.astype(np.float32)

    def end_drag(self) -> None:
        self.drag_vertex_idx[None] = -1
        self.drag_force[None] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_constraint_mode(self, mode: ConstraintMode) -> None:
        self.config.constraint_mode = mode
        self._apply_constraint_mode(self._rest_positions_np, mode)
        self.end_drag()

    def reset_state(self) -> None:
        self.x.from_numpy(self._rest_positions_np)
        self.v.fill(0.0)
        self.f.fill(0.0)
        self.end_drag()
        self._sim_time = 0.0

    def get_sim_time(self) -> float:
        return self._sim_time

    def advance_sim_time(self, dt: float) -> None:
        self._sim_time += dt

    def set_collision_world(self, world: CollisionWorld | None) -> None:
        self.collision_world = world

    def set_drag_params(self, stiffness: float, damping: float) -> None:
        self.drag_stiffness = stiffness
        self.drag_damping = damping

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

    # --- NumPy bridge for implicit solver ---
    def get_positions_numpy(self) -> np.ndarray:
        return self.x.to_numpy()

    def set_positions_numpy(self, x_np: np.ndarray) -> None:
        self.x.from_numpy(x_np.astype(np.float32))

    def get_velocities_numpy(self) -> np.ndarray:
        return self.v.to_numpy()

    def set_velocities_numpy(self, v_np: np.ndarray) -> None:
        self.v.from_numpy(v_np.astype(np.float32))

    def get_masses_numpy(self) -> np.ndarray:
        return self.mass.to_numpy()

    def get_fixed_numpy(self) -> np.ndarray:
        return self.fixed.to_numpy().astype(bool)

    def evaluate_internal_force_numpy(self, material_type: int, mu: float, lmbda: float) -> np.ndarray:
        self.clear_forces()
        self.accumulate_internal_forces(material_type, mu, lmbda)
        return self.f.to_numpy()

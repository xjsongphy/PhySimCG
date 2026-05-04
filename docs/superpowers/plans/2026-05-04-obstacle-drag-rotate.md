# Obstacle Drag & Rotate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement unified obstacle interaction (LMB move, RMB rotate) with ray-cast picking for both sphere and box obstacles, including instanced mesh rendering for boxes.

**Architecture:** Two-file change: `core.py` gets new obstacle fields (type, rotation quaternion, box size) and updated collision kernels; `gui.py` gets box mesh geometry, per-frame TRS transform building, mesh_instance rendering, ray-box picking, and RMB rotation logic.

**Tech Stack:** Python 3.12, Taichi, NumPy

---

### Task 1: Update OBSTACLES config and _create_sim

**Files:**
- Modify: `lab2/gui.py:30-39` (OBSTACLES dict)
- Modify: `lab2/gui.py:82-87` (_create_sim obstacle init)
- Modify: `lab2/core.py:128-134` (add new obstacle fields)

- [ ] **Step 1: Add new obstacle fields to FluidSimulator.__init__ in core.py**

After `self.obstacle_radius = ti.field(dtype=float, shape=(4,))` (line 133), add:

```python
self.obstacle_type = ti.field(dtype=int, shape=(4,))          # 0=sphere, 1=box
self.obstacle_rotation = ti.Vector.field(4, dtype=float, shape=(4,))  # quaternion (w,x,y,z)
self.obstacle_size = ti.Vector.field(3, dtype=float, shape=(4,))      # box half-extents
```

After `self.obstacle_count[None] = 0` (line 134), add:

```python
for o in range(4):
    self.obstacle_rotation[o] = [1.0, 0.0, 0.0, 0.0]
    self.obstacle_size[o] = [0.0, 0.0, 0.0]
```

- [ ] **Step 2: Update OBSTACLES dict in gui.py**

Replace `lab2/gui.py:31-38`:

```python
OBSTACLES = {
    "None":          [],
    "1 Sphere":      [("sphere", 0.06, None, (0.5, 0.25, 0.5))],
    "2 Spheres":     [("sphere", 0.05, None, (0.35, 0.25, 0.35)),
                      ("sphere", 0.05, None, (0.65, 0.25, 0.65))],
    "3 Spheres":     [("sphere", 0.045, None, (0.3, 0.25, 0.3)),
                      ("sphere", 0.045, None, (0.7, 0.25, 0.7)),
                      ("sphere", 0.045, None, (0.5, 0.25, 0.5))],
    "1 Big + 1 Small":[("sphere", 0.09, None, (0.5, 0.28, 0.5)),
                       ("sphere", 0.04, None, (0.7, 0.25, 0.3))],
    "Stirrer":       [("sphere", 0.08, None, (0.5, 0.25, 0.5))],
    "1 Box":         [("box", None, (0.08, 0.06, 0.08), (0.5, 0.25, 0.5))],
    "Sphere + Box":  [("sphere", 0.05, None, (0.35, 0.25, 0.35)),
                      ("box", None, (0.06, 0.08, 0.06), (0.65, 0.25, 0.65))],
}
```

- [ ] **Step 3: Update _create_sim obstacle initialization**

Replace `lab2/gui.py:82-87`:

```python
obs_list = OBSTACLES.get(obstacle_name, OBSTACLES["None"])
sim.obstacle_count[None] = len(obs_list)
for o, item in enumerate(obs_list):
    type_str, r_i, size_i, pos_i = item
    if type_str == "box":
        sim.obstacle_type[o] = 1
        sim.obstacle_size[o] = size_i
        sim.obstacle_radius[o] = 0.0
    else:
        sim.obstacle_type[o] = 0
        sim.obstacle_radius[o] = r_i
        sim.obstacle_size[o] = [0.0, 0.0, 0.0]
    sim.obstacle_pos[o] = pos_i
    sim.obstacle_rotation[o] = [1.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 4: Also update the obstacle-switch code in the GUI obstacle sub-window** (gui.py ~line 254-259)

Replace the obstacle switching block inside `with gui.sub_window("Obstacle", ...)`:

```python
for name in OBSTACLES:
    if g.button(name):
        obstacle_name = name
        obs_list = OBSTACLES[name]
        sim.obstacle_count[None] = len(obs_list)
        for o, item in enumerate(obs_list):
            type_str, r_i, size_i, pos_i = item
            if type_str == "box":
                sim.obstacle_type[o] = 1
                sim.obstacle_size[o] = size_i
                sim.obstacle_radius[o] = 0.0
            else:
                sim.obstacle_type[o] = 0
                sim.obstacle_radius[o] = r_i
                sim.obstacle_size[o] = [0.0, 0.0, 0.0]
            sim.obstacle_pos[o] = pos_i
            sim.obstacle_rotation[o] = [1.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 5: Commit**

```bash
git add lab2/core.py lab2/gui.py
git commit -m "feat: add obstacle type/rotation/size fields, update OBSTACLES config"
```

---

### Task 2: Update mark_obstacle_cells kernel for box type

**Files:**
- Modify: `lab2/core.py:340-361` (mark_obstacle_cells)

- [ ] **Step 1: Replace mark_obstacle_cells kernel**

Replace the existing kernel:

```python
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
                # conj(q) for inverse rotation
                cqw, cqx, cqy, cqz = qw, -qx, -qy, -qz
                # Bounding sphere radius for loop extent
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
                    # Transform to box local space: local = conj(q) * (0,v) * q
                    lx = cx - obs_pos[0]
                    ly = cy - obs_pos[1]
                    lz = cz - obs_pos[2]
                    # Step 1: t = cq * (0, lx, ly, lz)
                    tw = -cqx*lx - cqy*ly - cqz*lz
                    tx = cqw*lx + cqy*lz - cqz*ly
                    ty = cqw*ly - cqx*lz + cqz*lx
                    tz = cqw*lz + cqx*ly - cqy*lx
                    # Step 2: r = t * q
                    rx = tw*qx + tx*qw + ty*qz - tz*qy
                    ry = tw*qy - tx*qz + ty*qw + tz*qx
                    rz = tw*qz + tx*qy - ty*qx + tz*qw
                    if ti.abs(rx) < half_x and ti.abs(ry) < half_y and ti.abs(rz) < half_z:
                        self.cell_type[i, j, k] = 2
```

- [ ] **Step 2: Commit**

```bash
git add lab2/core.py
git commit -m "feat: update mark_obstacle_cells for box obstacles"
```

---

### Task 3: Update integrate_and_collide kernel for box collision

**Files:**
- Modify: `lab2/core.py:410-445` (integrate_and_collide)

- [ ] **Step 1: Replace the obstacle collision section in integrate_and_collide**

Replace lines 430-445 (obstacle collision within integrate_and_collide):

```python
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
                        # conj(q)
                        cqw, cqx, cqy, cqz = qw, -qx, -qy, -qz
                        # Transform particle to box local space: local = conj(q) * (0,v) * q
                        lx = self.pos[i][0] - obs_pos[0]
                        ly = self.pos[i][1] - obs_pos[1]
                        lz = self.pos[i][2] - obs_pos[2]
                        # Step 1: t = cq * (0, lx, ly, lz)
                        tw = -cqx*lx - cqy*ly - cqz*lz
                        tx = cqw*lx + cqy*lz - cqz*ly
                        ty = cqw*ly - cqx*lz + cqz*lx
                        tz = cqw*lz + cqx*ly - cqy*lx
                        # Step 2: r = t * q
                        rx = tw*qx + tx*qw + ty*qz - tz*qy
                        ry = tw*qy - tx*qz + ty*qw + tz*qx
                        rz = tw*qz + tx*qy - ty*qx + tz*qw
                        half_x, half_y, half_z = sz[0] + 0.01, sz[1] + 0.01, sz[2] + 0.01
                        push_x = ti.max(0.0, ti.abs(rx) - half_x)
                        push_y = ti.max(0.0, ti.abs(ry) - half_y)
                        push_z = ti.max(0.0, ti.abs(rz) - half_z)
                        if push_x > 0 or push_y > 0 or push_z > 0:
                            # Push direction in local space
                            sign_x = -1.0 if rx < 0 else 1.0
                            sign_y = -1.0 if ry < 0 else 1.0
                            sign_z = -1.0 if rz < 0 else 1.0
                            lpx, lpy, lpz = sign_x * push_x, sign_y * push_y, sign_z * push_z
                            # Transform local push to world: world = q * (0,local_push) * conj(q)
                            ptw = -qx*lpx - qy*lpy - qz*lpz
                            ptx = qw*lpx + qy*lpz - qz*lpy
                            pty = qw*lpy - qx*lpz + qz*lpx
                            ptz = qw*lpz + qx*lpy - qy*lpx
                            # Step 3: r = pt * cq  (but we only need vector part)
                            wpx = ptw*cqx + ptx*cqw + pty*cqz - ptz*cqy
                            wpy = ptw*cqy - ptx*cqz + pty*cqw + ptz*cqx
                            wpz = ptw*cqz + ptx*cqy - pty*cqx + ptz*cqw
                            self.pos[i][0] += wpx
                            self.pos[i][1] += wpy
                            self.pos[i][2] += wpz
                            # Reflect velocity
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
```

- [ ] **Step 2: Commit**

```bash
git add lab2/core.py
git commit -m "feat: update integrate_and_collide for box obstacle collision"
```

---

### Task 4: Add box mesh geometry and quaternion helpers to gui.py

**Files:**
- Modify: `lab2/gui.py`

- [ ] **Step 1: Add helper functions before run_gui()**

Add right after the OBSTACLES dict (after line 39):

```python
def _quat_mul(q1, q2):
    """Quaternion multiplication: q1 * q2.  Each is np.array([w, x, y, z])."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

def _quat_normalize(q):
    """Normalize a quaternion."""
    n = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n > 1e-12:
        q = q / n
    return q

def _quat_angle_axis(angle, axis):
    """Quaternion from angle (rad) and axis (3D vector)."""
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32)

def _quat_to_matrix(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ], dtype=np.float32)

def _build_box_geometry():
    """Return (vertices_field, indices_field) for unit cube [-0.5,0.5]^3."""
    verts = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    indices = np.array([
        0,1,2, 0,2,3,  # -Z
        4,6,5, 4,7,6,  # +Z
        0,3,7, 0,7,4,  # -X
        1,5,6, 1,6,2,  # +X
        0,4,5, 0,5,1,  # -Y
        3,2,6, 3,6,7,  # +Y
    ], dtype=np.int32)
    vf = ti.Vector.field(3, dtype=float, shape=len(verts))
    vf.from_numpy(verts)
    inf = ti.field(dtype=int, shape=len(indices))
    inf.from_numpy(indices)
    return vf, inf
```

- [ ] **Step 2: Commit**

```bash
git add lab2/gui.py
git commit -m "feat: add box mesh geometry and quaternion math helpers"
```

---

### Task 5: Update pick detection and interaction in gui.py

**Files:**
- Modify: `lab2/gui.py`

- [ ] **Step 1: Add ray-box intersection helper before run_gui()**

After the helpers from Task 4, add:

```python
def _ray_box_intersection(ray_origin, ray_dir, box_pos, box_quat, half_size):
    """Return min t > 0 if ray hits OBB, else None.
    Transforms ray to box local space, then does slab-test AABB."""
    q_inv = np.array([box_quat[0], -box_quat[1], -box_quat[2], -box_quat[3]])
    # Transform ray origin to local space
    local_org = np.array(ray_origin) - np.array(box_pos)
    lo = np.array([local_org[0], local_org[1], local_org[2], 0.0])
    lo_rot = _quat_mul(_quat_mul(q_inv, lo), box_quat)
    local_origin = np.array([lo_rot[1], lo_rot[2], lo_rot[3]])
    # Transform ray dir to local space
    ld = np.array([ray_dir[0], ray_dir[1], ray_dir[2], 0.0])
    ld_rot = _quat_mul(_quat_mul(q_inv, ld), box_quat)
    local_dir = np.array([ld_rot[1], ld_rot[2], ld_rot[3]])
    # Slab test
    hs = np.array(half_size)
    tmin = -1e30
    tmax = 1e30
    for d in range(3):
        if abs(local_dir[d]) > 1e-12:
            inv_d = 1.0 / local_dir[d]
            t1 = (-hs[d] - local_origin[d]) * inv_d
            t2 = ( hs[d] - local_origin[d]) * inv_d
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        else:
            if local_origin[d] < -hs[d] or local_origin[d] > hs[d]:
                return None
    if tmin > tmax or tmax < 0:
        return None
    return max(tmin, 0.0)
```

- [ ] **Step 2: Replace the pick detection block in run_gui()**

Replace lines 343-363 (ray-sphere intersection + pick loop):

```python
        # Ray-sphere / ray-box intersection for obstacle picking
        obs_count = sim.obstacle_count[None]
        picked_obs = -1
        if obs_count > 0:
            best_t = 1e30
            for o in range(min(obs_count, 4)):
                if sim.obstacle_type[o] == 0:  # sphere
                    obs_r = sim.obstacle_radius[o]
                    if obs_r <= 0:
                        continue
                    pick_r = max(obs_r * 2.5, 0.04)
                    obs_c = np.array([sim.obstacle_pos[o][0], sim.obstacle_pos[o][1], sim.obstacle_pos[o][2]])
                    oc = cam_pos - obs_c
                    a = ray_dir.dot(ray_dir)
                    b = 2.0 * oc.dot(ray_dir)
                    c = oc.dot(oc) - pick_r * pick_r
                    disc = b * b - 4 * a * c
                    if disc >= 0:
                        t = (-b - np.sqrt(disc)) / (2 * a)
                        if 0 < t < best_t:
                            best_t = t
                            picked_obs = o
                else:  # box
                    sz = np.array([sim.obstacle_size[o][0], sim.obstacle_size[o][1], sim.obstacle_size[o][2]])
                    box_pos_np = np.array([sim.obstacle_pos[o][0], sim.obstacle_pos[o][1], sim.obstacle_pos[o][2]])
                    quat_np = np.array([sim.obstacle_rotation[o][0], sim.obstacle_rotation[o][1],
                                        sim.obstacle_rotation[o][2], sim.obstacle_rotation[o][3]])
                    pick_size = sz + 0.04  # slight pick margin
                    t = _ray_box_intersection(cam_pos, ray_dir, box_pos_np, quat_np, pick_size)
                    if t is not None and 0 < t < best_t:
                        best_t = t
                        picked_obs = o
```

- [ ] **Step 3: Replace LMB/RMB drag logic to support both types and add rotation**

Replace lines 366-399 (the obstacle dragging block inside `if prev_cursor_valid:`):

```python
        # Obstacle dragging (camera-relative)
        dragging_obs = False
        rotating_obs = False
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y

            mouse_moved = abs(dx_mouse) > 0.002 or abs(dy_mouse) > 0.002
            if not mouse_moved:
                dx_mouse = 0.0
                dy_mouse = 0.0

            move_scale = 1.5

            # LMB: drag obstacle in camera plane
            if lmb and not over_gui and picked_obs >= 0:
                dragging_obs = True
                old = np.array([sim.obstacle_pos[picked_obs][0],
                               sim.obstacle_pos[picked_obs][1],
                               sim.obstacle_pos[picked_obs][2]])
                new = old + right_cam * (dx_mouse * move_scale) + cam_up * (-dy_mouse * move_scale)
                new[0] = np.clip(new[0], 0.05, 0.95)
                new[1] = np.clip(new[1], 0.05, 0.95)
                new[2] = np.clip(new[2], 0.05, 0.95)
                sim.obstacle_pos[picked_obs] = new
                sim.obstacle_vel[picked_obs] = (new - old) / (max(current_dt, 1e-6) * 4)

            # RMB: rotate obstacle about camera up/right axes
            if rmb and not over_gui and picked_obs >= 0:
                rotating_obs = True
                rot_speed = 0.005
                q_current = np.array([sim.obstacle_rotation[picked_obs][0],
                                     sim.obstacle_rotation[picked_obs][1],
                                     sim.obstacle_rotation[picked_obs][2],
                                     sim.obstacle_rotation[picked_obs][3]])
                dq_up = _quat_angle_axis(dx_mouse * rot_speed, cam_up)
                dq_right = _quat_angle_axis(dy_mouse * rot_speed, right_cam)
                delta = _quat_mul(dq_up, dq_right)
                q_new = _quat_normalize(_quat_mul(delta, q_current))
                sim.obstacle_rotation[picked_obs] = q_new
```

- [ ] **Step 4: Update camera control condition**

Replace line 402:

```python
        if prev_cursor_valid and not dragging_obs and not rotating_obs and not over_gui:
```

- [ ] **Step 5: Commit**

```bash
git add lab2/gui.py
git commit -m "feat: add ray-box pick, RMB rotation, unified obstacle interaction"
```

---

### Task 6: Build per-frame transforms and render boxes

**Files:**
- Modify: `lab2/gui.py`

- [ ] **Step 1: Init box geometry inside run_gui()**

Add right after `box_field.from_numpy(box_pts)` (line 149):

```python
box_mesh_verts, box_mesh_indices = _build_box_geometry()
box_transforms = ti.Matrix.field(4, 4, dtype=float, shape=(4,))
```

- [ ] **Step 2: Build transforms before rendering obstacle section**

Replace the obstacle rendering block (lines 499-515):

```python
        # Obstacles
        obs_count = sim.obstacle_count[None]
        if obs_count > 0:
            box_instance_count = 0
            box_transforms.fill(np.eye(4, dtype=np.float32))
            for o in range(min(obs_count, 4)):
                if sim.obstacle_type[o] == 0:  # sphere
                    obs_r = sim.obstacle_radius[o]
                    if obs_r > 0:
                        obs_pos_field = ti.Vector.field(3, dtype=float, shape=(1,))
                        obs_pos_field[0] = sim.obstacle_pos[o]
                        color = (0.9, 0.8, 0.2) if o == picked_obs else (0.8, 0.3, 0.3)
                        scene.particles(obs_pos_field, radius=obs_r * 0.95, color=color)
                else:  # box
                    pos = np.array([sim.obstacle_pos[o][0], sim.obstacle_pos[o][1], sim.obstacle_pos[o][2]], dtype=np.float32)
                    q = np.array([sim.obstacle_rotation[o][0], sim.obstacle_rotation[o][1],
                                 sim.obstacle_rotation[o][2], sim.obstacle_rotation[o][3]], dtype=np.float32)
                    s = np.array([sim.obstacle_size[o][0], sim.obstacle_size[o][1], sim.obstacle_size[o][2]], dtype=np.float32)
                    R = _quat_to_matrix(q)
                    T = np.eye(4, dtype=np.float32)
                    T[0,0], T[1,1], T[2,2] = s[0]*2, s[1]*2, s[2]*2  # half-extent -> full scale
                    T[:3, :3] = R @ T[:3, :3]  # R * S
                    T[0, 3], T[1, 3], T[2, 3] = pos[0], pos[1], pos[2]
                    box_transforms[box_instance_count] = T
                    box_instance_count += 1
            if box_instance_count > 0:
                color = (0.9, 0.8, 0.2) if picked_obs >= 0 and sim.obstacle_type[picked_obs] == 1 else (0.8, 0.3, 0.3)
                scene.mesh_instance(box_mesh_verts, box_mesh_indices,
                                   transforms=box_transforms,
                                   color=color, two_sided=True,
                                   instance_count=box_instance_count)
```

- [ ] **Step 3: Commit**

```bash
git add lab2/gui.py
git commit -m "feat: render box obstacles with instanced mesh"
```

---

### Task 7: End-to-end test

- [ ] **Step 1: Run the application**

```bash
cd /Users/xjsongphy/Codes/PhySimCG && python -m lab2
```

- [ ] **Step 2: Verify these behaviors manually**

1. "1 Sphere" obstacle: LMB drag moves it, RMB rotates (visually same for sphere, no crash)
2. "1 Box" obstacle: visible as solid cube, LMB drag moves it, RMB drag rotates it visibly
3. "Sphere + Box": can independently pick and interact with each
4. Camera pan/orbit works when clicking empty space
5. All existing sphere obstacles work as before
6. Box obstacle correctly blocks fluid particles (collision working)
7. Obstacle switching preserves new fields (type/rotation/size)


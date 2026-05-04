# Obstacle Drag & Rotate — Design Spec

**Date:** 2026-05-04
**Status:** Approved

## Overview

Implement unified obstacle interaction: LMB drag to move, RMB drag to rotate, with ray-cast picking for both sphere and box obstacles. Add box obstacle type with instanced mesh rendering.

---

## 1. Data Model (core.py)

### New fields

```python
self.obstacle_type = ti.field(dtype=int, shape=(4,))           # 0=sphere, 1=box
self.obstacle_rotation = ti.Vector.field(4, dtype=float, shape=(4,))  # quaternion (w,x,y,z)
self.obstacle_size = ti.Vector.field(3, dtype=float, shape=(4,))      # box half-extents
```

- `obstacle_type`: `0` = sphere (default, backward-compatible), `1` = box
- `obstacle_rotation`: wxyz quaternion, initialized to `(1, 0, 0, 0)` (identity)
- `obstacle_size`: box half-extents per axis, ignored for spheres

### OBSTACLES config format

Each entry: `(type_str, radius_or_None, size_or_None, position_tuple)`

```python
OBSTACLES = {
    "None":          [],
    "1 Sphere":      [("sphere", 0.06, None, (0.5, 0.25, 0.5))],
    "2 Spheres":     [...],
    "3 Spheres":     [...],
    "1 Big + 1 Small": [...],
    "Stirrer":       [("sphere", 0.08, None, (0.5, 0.25, 0.5))],
    "1 Box":         [("box", None, (0.08, 0.06, 0.08), (0.5, 0.25, 0.5))],
    "Sphere + Box":  [("sphere", 0.05, None, (0.35, 0.25, 0.35)),
                      ("box", None, (0.06, 0.08, 0.06), (0.65, 0.25, 0.65))],
}
```

### `_create_sim` changes

When building obstacles, set `obstacle_type[o]`, `obstacle_rotation[o] = [1, 0, 0, 0]`, and for boxes set `obstacle_size[o]`.

---

## 2. Box Rendering (gui.py)

### Shared geometry (initialized once)

- 8 vertices of a unit cube `[-0.5, 0.5]^3`
- 36 indices (12 triangles, 2 per face × 6 faces)
- Stored in `ti.Vector.field(3, shape=8)` and `ti.field(int, shape=36)`

### Per-frame transform

For each box obstacle, build a 4×4 TRS matrix:
- Scale by `obstacle_size * 2` (half-extents → full size)
- Rotate by `obstacle_rotation` (quaternion → rotation matrix)
- Translate by `obstacle_pos`

Store in `ti.Matrix.field((4, 4), shape=(4,))`.

### Render call

```python
scene.mesh_instance(box_vertices, box_indices, transforms=box_transforms,
                    color=pick_color, two_sided=True)
```

Only pass non-zero transforms to avoid rendering unused slots.

### Sphere rendering — unchanged, continues to use `scene.particles()`.

---

## 3. Interaction Logic (gui.py)

### Mapping

| Condition | LMB | RMB |
|-----------|-----|-----|
| Ray hits obstacle | Move obstacle (camera plane) | Rotate obstacle |
| Ray hits nothing | Camera pan (unchanged) | Camera orbit (unchanged) |

Priority: obstacle interaction > camera control.

### Pick detection

- **Sphere**: existing ray-sphere intersection (unchanged).
- **Box**: transform ray to box local space via inverse TRS, then ray-AABB intersection against `[-0.5, 0.5]^3`.
- Select nearest intersection across all obstacles.

### Move (LMB) — existing logic, unchanged

```
new_pos = old_pos + cam_right * (dx * move_scale) + cam_up * (-dy * move_scale)
clamp new_pos to [0.05, 0.95]
set obstacle_vel = (new - old) / dt
```

### Rotate (RMB) — new

```python
delta_rot = angleAxis(dx * 0.005, cam_up) * angleAxis(dy * 0.005, cam_right)
obstacle_rotation[o] = normalize(delta_rot * obstacle_rotation[o])
```

Speed factor 0.005 matches vcx-sim lab1. `cam_up` and `cam_right` are camera-local axes.

### Collision with fluid (core.py)

The existing `mark_obstacle_cells` kernel needs updating for box obstacles:
- Sphere branch: existing signed-distance check (unchanged)
- Box branch: transform particle to box local space, check AABB distance

---

## 4. Files Changed

| File | Changes |
|------|---------|
| `core.py` | New obstacle fields, updated `mark_obstacle_cells` for box collision, updated `integrate_and_clamp` for box collision |
| `gui.py` | Box mesh geometry init, per-frame transform build, mesh_instance render, ray-box pick, RMB rotation logic, OBSTACLES config updated |

No new files.

---

## 5. Testing

- All existing OBSTACLES presets still work (sphere rendering unchanged)
- New "1 Box" obstacle: appears as a solid box, LMB moves it, RMB rotates it
- New "Sphere + Box": both pickable independently
- Camera still works when clicking empty space
- Box rotation is visually verifiable (box orientation changes with RMB drag)

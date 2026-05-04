import time
import numpy as np
import taichi as ti
from lab2.core import FluidSimulator, scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.apic import APICSimulator
from lab2.eulerian import EulerianSimulator
from collections import defaultdict


def _get_screen_resolution() -> tuple:
    import sys, ctypes
    if sys.platform == 'darwin':
        try:
            import ctypes.util
            cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreGraphics'))
            display = cg.CGMainDisplayID()
            return (cg.CGDisplayPixelsWide(display),
                    cg.CGDisplayPixelsHigh(display))
        except Exception:
            pass
    elif sys.platform == 'win32':
        try:
            user32 = ctypes.windll.user32
            return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        except Exception:
            pass
    return (1920, 1080)


SCENES = ["Dam Break", "Drop", "Double Dam"]
OBSTACLES = {
    "None":          [],
    "1 Sphere":      [("sphere", 0.06, None, (0.5, 0.07, 0.5))],
    "2 Spheres":     [("sphere", 0.05, None, (0.35, 0.06, 0.35)),
                      ("sphere", 0.05, None, (0.65, 0.06, 0.65))],
    "3 Spheres":     [("sphere", 0.045, None, (0.3, 0.05, 0.3)),
                      ("sphere", 0.045, None, (0.7, 0.05, 0.7)),
                      ("sphere", 0.045, None, (0.5, 0.05, 0.5))],
    "1 Big + 1 Small":[("sphere", 0.09, None, (0.5, 0.10, 0.5)),
                       ("sphere", 0.04, None, (0.7, 0.05, 0.3))],
    "Stirrer":       [("sphere", 0.08, None, (0.5, 0.09, 0.5))],
    "1 Box":         [("box", None, (0.08, 0.06, 0.08), (0.5, 0.07, 0.5))],
    "Sphere + Box":  [("sphere", 0.05, None, (0.35, 0.06, 0.35)),
                      ("box", None, (0.06, 0.08, 0.06), (0.65, 0.09, 0.65))],
}

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

class _Profiler:
    """Lightweight frame profiler. Only active when debug_mode is True."""
    def __init__(self):
        self._cum = defaultdict(lambda: [0.0, 0])  # name → [total_ms, count]

    def record(self, name: str, ms: float):
        e = self._cum[name]
        e[0] += ms
        e[1] += 1

    def snapshot_and_reset(self):
        out = []
        for name, (total, n) in sorted(self._cum.items()):
            out.append((name, total, n, total / max(n, 1)))
        self._cum.clear()
        return out

_profiler = _Profiler()

# B1: COLOR_MODES = ["Speed", "Density", "Uniform"]
RESOLUTIONS = {
    "Low (16)":  (16, 32, 16),
    "Med (24)":  (24, 48, 24),
    "High (32)": (32, 64, 32),
}


def _create_sim(scene_name, nx, ny, nz, obstacle_name, method="FLIP"):
    np_ = scene_particle_count(scene_name, nx)
    if method == "APIC":
        sim = APICSimulator(nx, ny, nz, np_)
    else:
        sim = FLIPSimulator(nx, ny, nz, np_)
    if scene_name == "Dam Break":
        sim.init_dam_break()
    elif scene_name == "Drop":
        sim.init_drop()
    elif scene_name == "Double Dam":
        sim.init_double_dam()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()
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
    return sim


def _make_box_edges(lo, hi, n=30):
    pts = []
    for t in np.linspace(0, 1, n):
        for a in [lo, hi]:
            pts.append([a, lo, lo + t * (hi - lo)])
            pts.append([a, hi, lo + t * (hi - lo)])
            pts.append([lo + t * (hi - lo), a, lo])
            pts.append([lo + t * (hi - lo), a, hi])
            pts.append([lo, lo + t * (hi - lo), a])
            pts.append([hi, lo + t * (hi - lo), a])
    return np.array(pts, dtype=np.float32)


def _ray_box_intersection(ray_origin, ray_dir, box_pos, box_quat, half_size):
    """Return min t > 0 if ray hits OBB, else None.
    Transforms ray to box local space, then does slab-test AABB."""
    q_inv = np.array([box_quat[0], -box_quat[1], -box_quat[2], -box_quat[3]])
    local_org = np.array(ray_origin) - np.array(box_pos)
    lo = np.array([local_org[0], local_org[1], local_org[2], 0.0])
    lo_rot = _quat_mul(_quat_mul(q_inv, lo), box_quat)
    local_origin = np.array([lo_rot[1], lo_rot[2], lo_rot[3]])
    ld = np.array([ray_dir[0], ray_dir[1], ray_dir[2], 0.0])
    ld_rot = _quat_mul(_quat_mul(q_inv, ld), box_quat)
    local_dir = np.array([ld_rot[1], ld_rot[2], ld_rot[3]])
    hs = np.array(half_size)
    tmin, tmax = -1e30, 1e30
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


def run_gui(
    sim: FluidSimulator,
    substep_fn,
    *,
    dt: float = 0.01,
    num_substeps: int = 2,
    flip_ratio: float = 0.95,
    gravity: float = -9.8,
    window_title: str = "FLIP Fluid Simulation",
    window_size: tuple = None,
    debug: bool = False,
    # Panel visibility flags
    show_obstacle: bool = False,
    show_resolution: bool = False,
    show_color: bool = True,
    show_flip: bool = True,
    show_solver: bool = True,
    sim_type: str = "FLIP",  # "FLIP", "APIC", or "Eulerian"
):
    if window_size is None:
        screen_w, screen_h = _get_screen_resolution()
        window_size = (int(screen_w * 0.9), int(screen_h * 0.9))
    window = ti.ui.Window(window_title, window_size)
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.15))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Save initial volume for V-t tracking
    sim.save_init_volume()

    # Camera state
    _init_cam_target = np.array([0.5, 0.4, 0.5], dtype=np.float32)
    _init_cam_yaw = -1.57
    _init_cam_pitch = 0.45
    _init_cam_dist = 1.8
    _init_cam_pos = np.array([0.5, 1.0, 2.2], dtype=np.float32)
    cam_target = _init_cam_target.copy()
    cam_yaw = _init_cam_yaw
    cam_pitch = _init_cam_pitch
    cam_dist = _init_cam_dist
    cam_pos = _init_cam_pos.copy()

    # Container box
    dx = sim.dx
    box_pts = _make_box_edges(0.0, 1.0, n=40)
    box_field = ti.Vector.field(3, dtype=float, shape=box_pts.shape[0])
    box_field.from_numpy(box_pts)

    # Box mesh geometry
    box_mesh_verts, box_mesh_indices = _build_box_geometry()
    box_transforms = ti.Matrix.field(4, 4, dtype=float, shape=(4,))

    # State
    paused = False
    current_dt = dt
    current_flip_ratio = flip_ratio
    current_gravity = gravity
    current_scene = SCENES[0]
    current_color_mode = "Speed"
    current_res_name = "Med (24)"
    obstacle_name = "None"
    use_cg = False
    _sim_type = sim_type  # "FLIP", "APIC", or "Eulerian"
    sim_method = _sim_type if _sim_type != "Eulerian" else "FLIP"
    move_scale = 1.5
    animate_obstacle = False
    sim_time = 0.0
    shaking = False
    shake_strength = 3.0
    shake_freq = 0.5

    # Mouse tracking
    prev_cursor_x, prev_cursor_y = 0.0, 0.0
    prev_cursor_valid = False
    prev_lmb = False
    drag_target = -1  # sticky obstacle pick — persists while LMB held

    # Deferred recreation + debug timers
    _recreate = None
    debug_mode = debug
    _prof_frame = 0
    _ms_frame = 0.0
    t_frame = time.perf_counter()
    _vol_history = []

    while window.running:
        if debug_mode:
            t_frame = time.perf_counter()

        # ==== Deferred simulator recreation ====
        if _recreate is not None:
            t0 = time.perf_counter()
            scene_name, nx, ny, nz, st = _recreate
            if nz is None:
                nz = sim.nz
            _sim_type = st
            if st == "Eulerian":
                current_scene = scene_name
                sim = EulerianSimulator(nx, ny, nz)
                sim.init_dam_break_density()
                sim.init_cell_types()
                def substep_fn(dt, flip_ratio, gravity, **kw):
                    sim.substep(dt=dt, gravity=gravity, **kw)
                    sim.update_render_points()
            else:
                current_scene = scene_name
                sim_method = st
                sim = _create_sim(scene_name, nx, ny, nz, obstacle_name, sim_method)
                substep_fn = sim.substep
            dx = sim.dx
            _recreate = None
            if debug_mode:
                _profiler.record("recreate", (time.perf_counter() - t0) * 1000)

        gui = window.get_gui()

        # --- GUI: Scene ---
        with gui.sub_window("Scene", 0.02, 0.02, 0.22, 0.28) as g:
            g.text("=== Scene ===")
            g.text(f"  Current: {current_scene}")
            for name in SCENES:
                if g.button(name):
                    _recreate = (name, sim.nx, sim.ny, sim.nz, _sim_type)
            if g.button("Shake: ON" if shaking else "Shake: OFF"):
                shaking = not shaking
            shake_strength = g.slider_float("Strength", shake_strength, 0.0, 20.0)
            shake_freq = g.slider_float("Freq", shake_freq, 0.1, 3.0)

        # --- GUI: Resolution ---
        if show_resolution:
            with gui.sub_window("Resolution", 0.26, 0.22, 0.14, 0.22) as g:
                g.text("=== Resolution ===")
                g.text(f"  Grid: {sim.nx}x{sim.ny}x{sim.nz}")
                for name, (nx, ny, nz) in RESOLUTIONS.items():
                    if g.button(name):
                        current_res_name = name
                        _recreate = (current_scene, nx, ny, nz, _sim_type)

        # --- GUI: Controls ---
        with gui.sub_window("Controls", 0.02, 0.32, 0.22, 0.36) as g:
            g.text("=== Simulation ===")
            current_dt = g.slider_float("dt", current_dt, 0.002, 0.02)
            if show_flip:
                current_flip_ratio = g.slider_float("flipRatio", current_flip_ratio, 0.0, 1.0)
            current_gravity = g.slider_float("gravity", current_gravity, -20.0, 0.0)
            move_scale = g.slider_float("dragSpeed", move_scale, 0.1, 10.0)
            if g.button("Pause / Resume"):
                paused = not paused
            if g.button("Reset Sim"):
                _recreate = (current_scene, sim.nx, sim.ny, sim.nz, _sim_type)
            if g.button("Reset Camera"):
                cam_target[:] = _init_cam_target
                cam_yaw = _init_cam_yaw
                cam_pitch = _init_cam_pitch
                cam_dist = _init_cam_dist
                cam_pos[:] = _init_cam_pos
            if show_solver:
                g.text(f"  Solver: {'CG' if use_cg else 'GS'}")
                if g.button("Toggle CG/GS"):
                    use_cg = not use_cg
            if show_flip:
                g.text(f"  Method: {_sim_type}")
                if g.button("Toggle FLIP/APIC"):
                    new_method = "APIC" if _sim_type == "FLIP" else "FLIP"
                    _sim_type = new_method
                    _recreate = (current_scene, sim.nx, sim.ny, sim.nz, _sim_type)

        # --- GUI: Obstacle ---
        if show_obstacle:
            with gui.sub_window("Obstacle", 0.02, 0.70, 0.22, 0.20) as g:
                g.text("=== Obstacle ===")
                g.text(f"  Current: {obstacle_name}")
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
                if sim.obstacle_count[None] > 0:
                    g.text("  LMB: move | RMB: rotate")
                if g.button("Animate ON" if not animate_obstacle else "Animate OFF"):
                    animate_obstacle = not animate_obstacle

        # --- GUI: Color ---
        if show_color:
            with gui.sub_window("Color", 0.40, 0.02, 0.12, 0.22) as g:
                g.text("=== Color ===")
                g.text(f"  Mode: {current_color_mode}")
                for mode in ["Speed", "Density", "Uniform"]:
                    if g.button(mode):
                        current_color_mode = mode

        # --- GUI: Debug toggle ---
        with gui.sub_window("Debug", 0.26, 0.02, 0.12, 0.08) as g:
                g.text("=== Debug ===")
                if g.button("Debug ON" if not debug_mode else "Debug OFF"):
                    debug_mode = not debug_mode

        # --- GUI: Debug timing ---
        if debug_mode:
            stuck = sim.debug_count_stuck()
            if stuck > 0:
                sim.debug_color_stuck_red()
            _vol_history.append(sim._fluid_vol_ratio[0])
            if len(_vol_history) > 200:
                _vol_history.pop(0)

            with gui.sub_window("Debug Timing", 0.26, 0.46, 0.14, 0.38) as g:
                g.text(f"Frame:    {_ms_frame:>6.1f} ms")
                g.text(f"FPS:      {1000.0 / max(_ms_frame, 0.01):.0f}")
                g.text(f"Grid:     {sim.nx}x{sim.ny}x{sim.nz}")
                g.text(f"Particles:{sim.num_particles}")
                g.text(f"STUCK:    {stuck}  {'!!' if stuck else ''}")
                g.text(f"Volume:   {sim._fluid_vol_ratio[0]:.3f}")
                g.text(f"Time:     {sim_time:.2f} s")
                # Mini V-t graph (text-based)
                g.text("--- V-t (volume ratio) ---")
                n_pts = len(_vol_history)
                if n_pts > 1:
                    step = max(1, n_pts // 20)
                    for idx in range(0, n_pts, step):
                        v = _vol_history[idx]
                        bar_len = int(v * 15)
                        bar = "#" * bar_len + "." * (15 - bar_len)
                        g.text(f"  |{bar}|{v:.2f}")
                g.text("--- avg per call (last ~1s) ---")
                snap = _profiler.snapshot_and_reset()
                for name, total, n, avg in snap[:6]:
                    g.text(f"  {name:.<12s}{avg:6.1f}ms x{n}")

        # ==== Camera & Obstacle Interaction ====
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)

        # GUI panels occupy x ≤ 0.40; block camera interaction there
        over_gui = (cx < 0.42)

        # Compute ray from camera through cursor
        forward = cam_target - cam_pos
        forward_n = forward / (np.linalg.norm(forward) + 1e-8)
        if abs(forward_n[1]) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])
        else:
            world_up = np.array([0.0, 1.0, 0.0])
        right_cam = np.cross(forward_n, world_up)
        right_cam = right_cam / (np.linalg.norm(right_cam) + 1e-8)
        cam_up = np.cross(right_cam, forward_n)

        fov = np.radians(60)
        aspect = window_size[0] / max(window_size[1], 1)
        half_h = cam_dist * np.tan(fov / 2)
        half_w = half_h * aspect

        ndc_x = 2.0 * cx - 1.0
        ndc_y = 1.0 - 2.0 * cy
        pt_on_plane = cam_target + ndc_x * half_w * right_cam + ndc_y * half_h * cam_up
        ray_dir = pt_on_plane - cam_pos
        ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)

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
                    pick_r = max(obs_r * 10.0, 0.12)
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
                    pick_size = sz + 0.08
                    t = _ray_box_intersection(cam_pos, ray_dir, box_pos_np, quat_np, pick_size)
                    if t is not None and 0 < t < best_t:
                        best_t = t
                        picked_obs = o

        # Obstacle interaction — sticky pick: latch onto obstacle on LMB press, drag until release
        if lmb and not prev_lmb:
            if picked_obs >= 0:
                drag_target = picked_obs
        elif not lmb:
            drag_target = -1

        dragging_obs = False
        rotating_obs = False
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y

            mouse_moved = abs(dx_mouse) > 0.002 or abs(dy_mouse) > 0.002
            if not mouse_moved:
                dx_mouse = 0.0
                dy_mouse = 0.0

            if drag_target >= 0:
                # LMB: drag obstacle in camera plane
                if lmb:
                    dragging_obs = True
                    old = np.array([sim.obstacle_pos[drag_target][0],
                                   sim.obstacle_pos[drag_target][1],
                                   sim.obstacle_pos[drag_target][2]])
                    new = old + right_cam * (-dx_mouse * move_scale) + cam_up * (dy_mouse * move_scale)
                    new[0] = np.clip(new[0], 0.05, 0.95)
                    new[1] = np.clip(new[1], 0.05, 0.95)
                    new[2] = np.clip(new[2], 0.05, 0.95)
                    sim.obstacle_pos[drag_target] = new
                    sim.obstacle_vel[drag_target] = (new - old) / (max(current_dt, 1e-6) * 4)

                # RMB: rotate obstacle about camera axes
                if rmb:
                    rotating_obs = True
                    rot_speed = 0.005
                    q_current = np.array([sim.obstacle_rotation[drag_target][0],
                                         sim.obstacle_rotation[drag_target][1],
                                         sim.obstacle_rotation[drag_target][2],
                                         sim.obstacle_rotation[drag_target][3]])
                    dq_up = _quat_angle_axis(dx_mouse * rot_speed, cam_up)
                    dq_right = _quat_angle_axis(dy_mouse * rot_speed, right_cam)
                    delta = _quat_mul(dq_up, dq_right)
                    q_new = _quat_normalize(_quat_mul(delta, q_current))
                    sim.obstacle_rotation[drag_target] = q_new

        # Camera controls (only when not interacting with obstacle)
        if prev_cursor_valid and not dragging_obs and not rotating_obs and not over_gui:
            if rmb:
                cam_yaw += dx_mouse * 3.0
                cam_pitch -= dy_mouse * 3.0
                cam_pitch = np.clip(cam_pitch, -1.55, 1.55)

            if lmb and not over_gui:
                pan_speed = cam_dist * 0.8
                pan_x = dx_mouse * pan_speed
                pan_y = dy_mouse * pan_speed
                forward_h = forward.copy()
                forward_h[1] = 0
                forward_h = forward_h / (np.linalg.norm(forward_h) + 1e-8)
                right_pan = np.array([forward_h[2], 0, -forward_h[0]])
                cam_target += right_pan * pan_x - np.array([0, pan_y, 0])

        # Zoom keys
        if window.is_pressed('r'):
            cam_dist = max(0.2, cam_dist - 0.05)
        if window.is_pressed('f'):
            cam_dist = min(10.0, cam_dist + 0.05)

        # Update camera position
        cam_pos[0] = cam_target[0] + cam_dist * np.cos(cam_pitch) * np.cos(cam_yaw)
        cam_pos[1] = cam_target[1] + cam_dist * np.sin(cam_pitch)
        cam_pos[2] = cam_target[2] + cam_dist * np.cos(cam_pitch) * np.sin(cam_yaw)

        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])

        prev_cursor_x, prev_cursor_y = cx, cy
        prev_cursor_valid = True
        prev_lmb = lmb

        # ==== Animate obstacle ====
        if animate_obstacle and sim.obstacle_count[None] > 0:
            t = sim_time
            old_pos = np.array([sim.obstacle_pos[0][0], sim.obstacle_pos[0][1], sim.obstacle_pos[0][2]])
            cx = 0.5 + 0.25 * np.sin(t * 2.0)
            cz = 0.5 + 0.25 * np.cos(t * 2.0)
            cy = old_pos[1]
            new_pos = np.array([cx, cy, cz])
            vel = (new_pos - old_pos) / max(current_dt, 1e-6)
            sim.obstacle_pos[0] = new_pos
            sim.obstacle_vel[0] = vel

        # ==== Simulation ====
        if debug_mode:
            t_sim = time.perf_counter()
        if not paused:
            if shaking:
                freq = 1.2
                gx = shake_strength * np.sin(sim_time * shake_freq * 2 * np.pi)
                gz = shake_strength * np.cos(sim_time * shake_freq * 2 * np.pi) * 0.7
                sim.apply_horizontal_impulse(gx * current_dt, gz * current_dt)
            if sim.obstacle_count[None] > 0:
                sim.mark_obstacle_cells()
            for _ in range(num_substeps):
                substep_fn(
                    dt=current_dt,
                    flip_ratio=current_flip_ratio,
                    gravity=current_gravity,
                )
            sim_time += current_dt * num_substeps
        if debug_mode:
            _profiler.record("sim", (time.perf_counter() - t_sim) * 1000)

        # ==== Colors ====
        if debug_mode:
            t_col = time.perf_counter()
        if current_color_mode == "Speed":
            sim.update_default_colors()
        elif current_color_mode == "Density":
            # Smooth density color updates - only update if density changed significantly
            if hasattr(sim, 'update_colors_by_density'):
                sim.update_colors_by_density()
        else:
            sim.update_colors_uniform()
        if debug_mode:
            _profiler.record("color", (time.perf_counter() - t_col) * 1000)

        # ==== Render ====
        if debug_mode:
            t_rend = time.perf_counter()
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.7, 0.7, 0.75))

        # Container wireframe
        scene.particles(box_field, radius=dx * 0.05, color=(0.5, 0.5, 0.5))

        # Fluid
        scene.particles(
            sim.pos,
            radius=dx * 0.15,
            per_vertex_color=sim.color,
        )

        # Obstacles
        obs_count = sim.obstacle_count[None]
        if obs_count > 0:
            box_instance_count = 0
            box_transforms.fill([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            highlight_obs = drag_target if drag_target >= 0 else picked_obs
            for o in range(min(obs_count, 4)):
                if sim.obstacle_type[o] == 0:  # sphere
                    obs_r = sim.obstacle_radius[o]
                    if obs_r > 0:
                        obs_pos_field = ti.Vector.field(3, dtype=float, shape=(1,))
                        obs_pos_field[0] = sim.obstacle_pos[o]
                        color = (0.9, 0.8, 0.2) if o == highlight_obs else (0.8, 0.3, 0.3)
                        scene.particles(obs_pos_field, radius=obs_r * 0.95, color=color)
                else:  # box
                    pos = np.array([sim.obstacle_pos[o][0], sim.obstacle_pos[o][1], sim.obstacle_pos[o][2]], dtype=np.float32)
                    q = np.array([sim.obstacle_rotation[o][0], sim.obstacle_rotation[o][1],
                                 sim.obstacle_rotation[o][2], sim.obstacle_rotation[o][3]], dtype=np.float32)
                    s = np.array([sim.obstacle_size[o][0], sim.obstacle_size[o][1], sim.obstacle_size[o][2]], dtype=np.float32)
                    R = _quat_to_matrix(q)
                    T = np.eye(4, dtype=np.float32)
                    T[0, 0] = s[0] * 2
                    T[1, 1] = s[1] * 2
                    T[2, 2] = s[2] * 2
                    T[:3, :3] = R @ T[:3, :3]
                    T[0, 3] = pos[0]
                    T[1, 3] = pos[1]
                    T[2, 3] = pos[2]
                    box_transforms[box_instance_count] = T
                    box_instance_count += 1
            if box_instance_count > 0:
                color = (0.9, 0.8, 0.2) if highlight_obs >= 0 and sim.obstacle_type[highlight_obs] == 1 else (0.8, 0.3, 0.3)
                scene.mesh_instance(box_mesh_verts, box_mesh_indices,
                                   transforms=box_transforms,
                                   color=color, two_sided=True,
                                   instance_count=box_instance_count)

        canvas.scene(scene)
        window.show()
        if debug_mode:
            _profiler.record("render", (time.perf_counter() - t_rend) * 1000)
            _ms_frame = (time.perf_counter() - t_frame) * 1000

import time
import numpy as np
import taichi as ti
from lab2.core import FluidSimulator, scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.apic import APICSimulator
from collections import defaultdict


def _get_screen_resolution() -> tuple:
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return (width, height)
    except Exception:
        return (1920, 1080)


SCENES = ["Dam Break", "Drop", "Double Dam"]
OBSTACLES = {
    "None":          [],
    "1 Sphere":      [(0.06, (0.5, 0.5, 0.5))],
    "2 Spheres":     [(0.05, (0.35, 0.6, 0.35)), (0.05, (0.65, 0.4, 0.65))],
    "3 Spheres":     [(0.045, (0.3, 0.65, 0.3)), (0.045, (0.7, 0.45, 0.7)),
                      (0.045, (0.5, 0.55, 0.5))],
    "1 Big + 1 Small":[(0.09, (0.5, 0.55, 0.5)), (0.04, (0.7, 0.5, 0.3))],
}
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
    for o, (r_i, pos_i) in enumerate(obs_list):
        sim.obstacle_radius[o] = r_i
        sim.obstacle_pos[o] = pos_i
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
):
    if window_size is None:
        screen_w, screen_h = _get_screen_resolution()
        window_size = (int(screen_w * 0.9), int(screen_h * 0.9))
    window = ti.ui.Window(window_title, window_size)
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.15))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Camera state (OrbitCamera-style)
    cam_target = np.array([0.5, 0.4, 0.5], dtype=np.float32)
    cam_yaw = -1.57     # radians, horizontal orbit angle
    cam_pitch = 0.45    # radians, vertical orbit angle
    cam_dist = 1.8      # distance from target
    cam_pos = np.array([0.5, 1.0, 2.2], dtype=np.float32)

    # Container box
    dx = sim.dx
    box_pts = _make_box_edges(0.0, 1.0, n=40)
    box_field = ti.Vector.field(3, dtype=float, shape=box_pts.shape[0])
    box_field.from_numpy(box_pts)

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
    sim_method = "FLIP"  # "FLIP" or "APIC"

    # Mouse tracking
    prev_cursor_x, prev_cursor_y = 0.0, 0.0
    prev_cursor_valid = False

    # Deferred recreation + debug timers
    _recreate = None
    debug_mode = debug
    _prof_frame = 0
    _ms_frame = 0.0

    while window.running:
        if debug_mode:
            t_frame = time.perf_counter()

        # ==== Deferred simulator recreation ====
        if _recreate is not None:
            t0 = time.perf_counter()
            scene_name, nx, ny, nz = _recreate
            if nz is None:
                nz = sim.nz
            current_scene = scene_name
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
                    _recreate = (name, sim.nx, sim.ny, sim.nz)
            if g.button("Perturb!"):
                sim.apply_perturbation(0.5, 0.4, 0.5, 6.0)

        # --- GUI: Resolution ---
        with gui.sub_window("Resolution", 0.26, 0.22, 0.14, 0.22) as g:
            g.text("=== Resolution ===")
            g.text(f"  Grid: {sim.nx}x{sim.ny}x{sim.nz}")
            for name, (nx, ny, nz) in RESOLUTIONS.items():
                if g.button(name):
                    current_res_name = name
                    _recreate = (current_scene, nx, ny, nz)

        # --- GUI: Controls ---
        with gui.sub_window("Controls", 0.02, 0.32, 0.22, 0.32) as g:
            g.text("=== Simulation ===")
            current_dt = g.slider_float("dt", current_dt, 0.001, 0.03)
            current_flip_ratio = g.slider_float("flipRatio", current_flip_ratio, 0.0, 1.0)
            current_gravity = g.slider_float("gravity", current_gravity, -20.0, 0.0)
            if g.button("Pause / Resume"):
                paused = not paused
            g.text(f"  Solver: {'CG' if use_cg else 'GS'}")
            if g.button("Toggle CG/GS"):
                use_cg = not use_cg

        # --- GUI: Obstacle ---
        with gui.sub_window("Obstacle", 0.02, 0.62, 0.22, 0.25) as g:
            g.text("=== Obstacle ===")
            g.text(f"  Current: {obstacle_name}")
            for name in OBSTACLES:
                if g.button(name):
                    obstacle_name = name
                    obs_list = OBSTACLES[name]
                    sim.obstacle_count[None] = len(obs_list)
                    for o, (r_i, pos_i) in enumerate(obs_list):
                        sim.obstacle_radius[o] = r_i
                        sim.obstacle_pos[o] = pos_i
            if sim.obstacle_count[None] > 0:
                g.text("  MMB: drag 1st obs")

        # --- GUI: Color ---
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
            with gui.sub_window("Debug Timing", 0.26, 0.46, 0.14, 0.28) as g:
                g.text(f"Frame:    {_ms_frame:>6.1f} ms")
                g.text(f"FPS:      {1000.0 / max(_ms_frame, 0.01):.0f}")
                g.text(f"Grid:     {sim.nx}x{sim.ny}x{sim.nz}")
                g.text(f"Particles:{sim.num_particles}")
                g.text("--- avg per call (last ~1s) ---")
                snap = _profiler.snapshot_and_reset()
                for name, total, n, avg in snap[:8]:
                    g.text(f"  {name:.<12s}{avg:6.1f}ms x{n}")

        # ==== Camera & Obstacle Interaction ====
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)

        # Skip camera controls when cursor is over GUI panels
        over_gui = (cx < 0.54 and cy < 0.88)

        # Compute ray from camera through cursor
        forward = cam_target - cam_pos
        forward_n = forward / (np.linalg.norm(forward) + 1e-8)
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

        # Ray-sphere intersection for obstacle picking
        obs_count = sim.obstacle_count[None]
        picked_obs = -1
        if obs_count > 0:
            best_t = 1e30
            for o in range(min(obs_count, 4)):
                obs_r = sim.obstacle_radius[o]
                if obs_r <= 0:
                    continue
                oc = cam_pos - np.array([sim.obstacle_pos[o][0], sim.obstacle_pos[o][1], sim.obstacle_pos[o][2]])
                a = ray_dir.dot(ray_dir)
                b = 2.0 * oc.dot(ray_dir)
                c = oc.dot(oc) - obs_r * obs_r
                disc = b * b - 4 * a * c
                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / (2 * a)
                    if 0 < t < best_t:
                        best_t = t
                        picked_obs = o

        # LMB: drag obstacle (if picked) or pan camera
        dragging_obs = False
        if lmb and not over_gui and picked_obs >= 0:
            dragging_obs = True
            # Drag obstacle along the plane perpendicular to ray at hit point
            plane_y = sim.obstacle_pos[picked_obs][1]
            if abs(ray_dir[1]) > 1e-8:
                t_hit = (plane_y - cam_pos[1]) / ray_dir[1]
                if t_hit > 0:
                    hit = cam_pos + t_hit * ray_dir
                    hit[0] = np.clip(hit[0], 0.05, 0.95)
                    hit[2] = np.clip(hit[2], 0.05, 0.95)
                    hit[1] = plane_y
                    old_pos = np.array([
                        sim.obstacle_pos[picked_obs][0],
                        sim.obstacle_pos[picked_obs][1],
                        sim.obstacle_pos[picked_obs][2],
                    ])
                    vel_est = (hit - old_pos) / max(current_dt, 1e-6)
                    sim.obstacle_pos[picked_obs] = hit
                    sim.obstacle_vel[picked_obs] = vel_est

        # Camera controls
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y

            # RMB: rotate camera direction (orbit)
            if rmb and not over_gui:
                cam_yaw += dx_mouse * 3.0
                cam_pitch -= dy_mouse * 3.0
                cam_pitch = np.clip(cam_pitch, 0.05, 1.5)

            # LMB (no obstacle hit): pan camera position
            if lmb and not over_gui and not dragging_obs:
                pan_speed = cam_dist * 0.8
                pan_x = dx_mouse * pan_speed
                pan_y = -dy_mouse * pan_speed
                forward_h = forward.copy()
                forward_h[1] = 0
                forward_h = forward_h / (np.linalg.norm(forward_h) + 1e-8)
                right_pan = np.array([forward_h[2], 0, -forward_h[0]])
                cam_target += right_pan * pan_x + np.array([0, pan_y, 0])

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

        # ==== Simulation ====
        if debug_mode:
            t_sim = time.perf_counter()
        if not paused:
            if sim.obstacle_count[None] > 0:
                sim.mark_obstacle_cells()
            for _ in range(num_substeps):
                substep_fn(
                    dt=current_dt,
                    flip_ratio=current_flip_ratio,
                    gravity=current_gravity,
                    use_cg=use_cg,
                )
        if debug_mode:
            _profiler.record("sim", (time.perf_counter() - t_sim) * 1000)

        # ==== Colors ====
        if debug_mode:
            t_col = time.perf_counter()
        if current_color_mode == "Speed":
            sim.update_default_colors()
        elif current_color_mode == "Density":
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
        scene.particles(box_field, radius=dx * 0.08, color=(0.5, 0.5, 0.5))

        # Fluid
        scene.particles(
            sim.pos,
            radius=dx * 0.25,
            per_vertex_color=sim.color,
        )

        # Obstacles
        obs_count = sim.obstacle_count[None]
        if obs_count > 0:
            for o in range(min(obs_count, 4)):
                obs_r = sim.obstacle_radius[o]
                if obs_r > 0:
                    obs_pos_field = ti.Vector.field(3, dtype=float, shape=(1,))
                    obs_pos_field[0] = sim.obstacle_pos[o]
                    scene.particles(
                        obs_pos_field,
                        radius=obs_r * 0.95,
                        color=(0.8, 0.3, 0.3),
                    )

        canvas.scene(scene)
        window.show()
        if debug_mode:
            _profiler.record("render", (time.perf_counter() - t_rend) * 1000)
            _ms_frame = (time.perf_counter() - t_frame) * 1000

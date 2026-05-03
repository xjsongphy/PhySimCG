import time
import numpy as np
import taichi as ti
from lab2.core import FluidSimulator, scene_particle_count
from lab2.flip import FLIPSimulator


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
    "None":        (0.0, (0.5, 0.5, 0.5)),
    "1 Sphere":    (0.06, (0.5, 0.5, 0.5)),
    "1 Big Sphere":(0.10, (0.5, 0.5, 0.5)),
}
COLOR_MODES = ["Speed", "Density", "Uniform"]
RESOLUTIONS = {
    "Low (16)":  (16, 32, 16),
    "Med (24)":  (24, 48, 24),
    "High (32)": (32, 64, 32),
}


def _create_sim(scene_name, nx, ny, nz, obstacle_name):
    np_ = scene_particle_count(scene_name, nx)
    sim = FLIPSimulator(nx, ny, nz, np_)
    if scene_name == "Dam Break":
        sim.init_dam_break()
    elif scene_name == "Drop":
        sim.init_drop()
    elif scene_name == "Double Dam":
        sim.init_double_dam()
    sim.init_cell_types()
    sim.init_colors()
    sim.update_particle_density()
    sim.store_initial_density()
    obs_r, obs_pos = OBSTACLES.get(obstacle_name, OBSTACLES["None"])
    if obs_r > 0:
        sim.obstacle_radius[None] = obs_r
        sim.obstacle_pos[0] = obs_pos
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
    current_color_mode = COLOR_MODES[0]
    current_res_name = "Med (24)"
    obstacle_name = "None"

    # Mouse tracking
    prev_cursor_x, prev_cursor_y = 0.0, 0.0
    prev_cursor_valid = False

    # Deferred recreation + debug timers
    _recreate = None
    debug_mode = False
    _ms_recreate = 0.0
    _ms_substep = 0.0
    _ms_render = 0.0
    _ms_frame = 0.0

    while window.running:
        t_frame = time.perf_counter()

        # ==== Deferred simulator recreation ====
        if _recreate is not None:
            t0 = time.perf_counter()
            scene_name, nx, ny, nz = _recreate
            if nz is None:
                nz = sim.nz
            current_scene = scene_name
            sim = _create_sim(scene_name, nx, ny, nz, obstacle_name)
            substep_fn = sim.substep
            dx = sim.dx
            _recreate = None
            _ms_recreate = (time.perf_counter() - t0) * 1000

        gui = window.get_gui()

        # --- GUI: Scene ---
        with gui.sub_window("Scene", 0.02, 0.02, 0.22, 0.28) as g:
            g.text("=== Scene ===")
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
        with gui.sub_window("Controls", 0.02, 0.32, 0.22, 0.28) as g:
            g.text("=== Simulation ===")
            current_dt = g.slider_float("dt", current_dt, 0.001, 0.03)
            current_flip_ratio = g.slider_float("flipRatio", current_flip_ratio, 0.0, 1.0)
            current_gravity = g.slider_float("gravity", current_gravity, -20.0, 0.0)
            if g.button("Pause / Resume"):
                paused = not paused

        # --- GUI: Obstacle ---
        with gui.sub_window("Obstacle", 0.02, 0.62, 0.22, 0.22) as g:
            g.text("=== Obstacle ===")
            for name in OBSTACLES:
                if g.button(name):
                    obstacle_name = name
                    obs_r, obs_pos = OBSTACLES[name]
                    if obs_r > 0:
                        sim.obstacle_radius[None] = obs_r
                        sim.obstacle_pos[0] = obs_pos
                    else:
                        sim.obstacle_radius[None] = 0.0

        # --- GUI: Color ---
        with gui.sub_window("Color", 0.26, 0.02, 0.14, 0.18) as g:
            g.text("=== Color ===")
            for mode in COLOR_MODES:
                if g.button(mode):
                    current_color_mode = mode
            if g.button("Debug ON" if not debug_mode else "Debug OFF"):
                debug_mode = not debug_mode

        # --- GUI: Debug timing ---
        if debug_mode:
            with gui.sub_window("Debug Timing", 0.26, 0.46, 0.14, 0.20) as g:
                g.text(f"Recreate: {_ms_recreate:>6.1f} ms")
                g.text(f"Substeps: {_ms_substep:>6.1f} ms")
                g.text(f"Render:   {_ms_render:>6.1f} ms")
                g.text(f"Frame:    {_ms_frame:>6.1f} ms")
                g.text(f"FPS:      {1000.0 / max(_ms_frame, 0.01):.0f}")
                g.text(f"Particles:{sim.num_particles}")
                g.text(f"Grid:     {sim.nx}x{sim.ny}x{sim.nz}")

        # ==== Camera & Obstacle Interaction ====
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)

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

        # Camera orbit (LMB rotate, RMB pan)
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y

            if lmb:
                cam_yaw += dx_mouse * 3.0
                cam_pitch -= dy_mouse * 3.0
                cam_pitch = np.clip(cam_pitch, 0.05, 1.5)

            # RMB: Pan
            if rmb:
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
        t_sim = time.perf_counter()
        if not paused:
            obs_r = sim.obstacle_radius[None]
            if obs_r > 0:
                sim.mark_obstacle_cells()
            for _ in range(num_substeps):
                substep_fn(
                    dt=current_dt,
                    flip_ratio=current_flip_ratio,
                    gravity=current_gravity,
                )

        # ==== Colors ====
        if current_color_mode == "Speed":
            sim.update_default_colors()
        elif current_color_mode == "Density":
            sim.update_colors_by_density()
        else:
            sim.update_colors_uniform()

        # ==== Render ====
        t_render = time.perf_counter()
        _ms_substep = (t_render - t_sim) * 1000
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.7, 0.7, 0.75))

        # Container wireframe
        scene.particles(box_field, radius=dx * 0.08, color=(0.5, 0.5, 0.5))

        # Fluid
        scene.particles(
            sim.pos,
            radius=dx * 0.35,
            per_vertex_color=sim.color,
        )

        # Obstacle
        obs_r = sim.obstacle_radius[None]
        if obs_r > 0:
            scene.particles(
                sim.obstacle_pos,
                radius=obs_r * 0.95,
                color=(0.8, 0.3, 0.3),
            )

        canvas.scene(scene)
        window.show()
        t_end = time.perf_counter()
        _ms_render = (t_end - t_render) * 1000
        _ms_frame = (t_end - t_frame) * 1000

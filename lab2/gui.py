import numpy as np
import taichi as ti
from lab2.core import FluidSimulator


SCENES = ["Dam Break", "Drop", "Double Dam"]
COLOR_MODES = ["Speed", "Density", "Uniform"]


def _reset_scene(sim: FluidSimulator, scene_name: str):
    if scene_name == "Dam Break":
        sim.init_dam_break()
    elif scene_name == "Drop":
        sim.init_drop()
    elif scene_name == "Double Dam":
        sim.init_double_dam()
    sim.init_cell_types()
    sim.update_particle_density()
    sim.store_initial_density()


def _make_box_edges(lo, hi, n=30):
    """Create points along 12 edges of a box [lo, hi] for rendering."""
    pts = []
    for t in np.linspace(0, 1, n):
        for a in [lo, hi]:
            # edges along x
            pts.append([a, lo, lo + t * (hi - lo)])
            pts.append([a, hi, lo + t * (hi - lo)])
            # edges along y
            pts.append([lo + t * (hi - lo), a, lo])
            pts.append([lo + t * (hi - lo), a, hi])
            # edges along z
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
):
    window = ti.ui.Window(window_title, (1920, 1080))
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.15))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 1.0, 2.2)
    camera.lookat(0.5, 0.4, 0.5)

    # Container box visualization
    dx = sim.dx
    box_pts = _make_box_edges(dx, 1.0 - dx, n=40)
    box_field = ti.Vector.field(3, dtype=float, shape=box_pts.shape[0])
    box_field.from_numpy(box_pts)

    # State
    paused = False
    current_dt = dt
    current_flip_ratio = flip_ratio
    current_gravity = gravity
    current_scene = SCENES[0]
    current_color_mode = COLOR_MODES[0]
    obstacle_enabled = False
    obstacle_radius = 0.06

    while window.running:
        # --- GUI Controls ---
        gui = window.get_gui()

        with gui.sub_window("Scene", 0.02, 0.02, 0.22, 0.28) as g:
            g.text("=== Scene ===")
            for name in SCENES:
                if g.button(name):
                    current_scene = name
                    _reset_scene(sim, name)
            if g.button("Perturb!"):
                sim.apply_perturbation(0.5, 0.4, 0.5, 6.0)

        with gui.sub_window("Controls", 0.02, 0.32, 0.22, 0.28) as g:
            g.text("=== Simulation ===")
            current_dt = g.slider_float("dt", current_dt, 0.001, 0.03)
            current_flip_ratio = g.slider_float("flipRatio", current_flip_ratio, 0.0, 1.0)
            current_gravity = g.slider_float("gravity", current_gravity, -20.0, 0.0)
            if g.button("Pause / Resume"):
                paused = not paused

        with gui.sub_window("Obstacle", 0.02, 0.62, 0.22, 0.22) as g:
            g.text("=== Obstacle ===")
            if g.button("Toggle Obstacle"):
                obstacle_enabled = not obstacle_enabled
                if obstacle_enabled:
                    sim.obstacle_radius[None] = obstacle_radius
                    sim.obstacle_pos[0] = [0.5, 0.5, 0.5]
                else:
                    sim.obstacle_radius[None] = 0.0
            obstacle_radius = g.slider_float("radius", obstacle_radius, 0.03, 0.15)
            if obstacle_enabled:
                sim.obstacle_radius[None] = obstacle_radius
            g.text(f"  {'ON' if obstacle_enabled else 'OFF'} (LMB drag)")

        with gui.sub_window("Color", 0.26, 0.02, 0.14, 0.18) as g:
            g.text("=== Color ===")
            for mode in COLOR_MODES:
                if g.button(mode):
                    current_color_mode = mode

        # --- Mouse interaction ---
        if obstacle_enabled and window.is_pressed(ti.ui.LMB):
            cursor_x, cursor_y = window.get_cursor_pos()
            obs_y = sim.obstacle_pos[0][1]
            new_pos = np.array([
                max(0.05, min(0.95, cursor_x)),
                obs_y,
                max(0.05, min(0.95, cursor_y)),
            ], dtype=np.float32)
            old_pos = sim.obstacle_pos[0].to_numpy()
            sim.obstacle_vel[0] = (new_pos - old_pos) / max(current_dt, 1e-6)
            sim.obstacle_pos[0] = new_pos
        else:
            sim.obstacle_vel[0] = [0.0, 0.0, 0.0]

        # --- Simulation step ---
        if not paused:
            if obstacle_enabled:
                sim.mark_obstacle_cells()

            for _ in range(num_substeps):
                substep_fn(
                    dt=current_dt,
                    flip_ratio=current_flip_ratio,
                    gravity=current_gravity,
                )

        # --- Update colors ---
        if current_color_mode == "Speed":
            sim.update_default_colors()
        elif current_color_mode == "Density":
            sim.update_colors_by_density()
        else:
            sim.update_colors_uniform()

        # --- Render ---
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.4, 0.4, 0.5))

        # Container box (wireframe via tiny particles)
        scene.particles(box_field, radius=dx * 0.08, color=(0.5, 0.5, 0.5))

        # Fluid particles
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

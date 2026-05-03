import taichi as ti
from lab2.core import FluidSimulator


def run_gui(
    sim: FluidSimulator,
    substep_fn,
    *,
    dt: float = 0.02,
    num_substeps: int = 2,
    flip_ratio: float = 0.95,
    gravity: float = -9.8,
    window_title: str = "FLIP Fluid Simulation",
):
    """Run GGUI rendering loop with interactive sliders."""
    window = ti.ui.Window(window_title, (1280, 720))
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.15))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 1.2, 2.0)
    camera.lookat(0.5, 0.4, 0.5)

    # State
    paused = False
    current_dt = dt
    current_flip_ratio = flip_ratio
    current_gravity = gravity

    while window.running:
        # GUI controls
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.3, 0.35) as g:
            g.text(f"Particles: {sim.num_particles}")
            g.text(f"Grid: {sim.nx}x{sim.ny}x{sim.nz}")
            current_dt = g.slider_float("dt", current_dt, 0.001, 0.05)
            current_flip_ratio = g.slider_float("flipRatio", current_flip_ratio, 0.0, 1.0)
            current_gravity = g.slider_float("gravity", current_gravity, -20.0, 0.0)
            g.text(f"  flipRatio={current_flip_ratio:.2f}  (0=PIC, 1=FLIP)")
            if g.button("Pause / Resume"):
                paused = not paused
            if g.button("Reset"):
                sim.init_dam_break()
                sim.init_cell_types()
                sim.update_particle_density()
                sim.store_initial_density()

        # Simulation step
        if not paused:
            for _ in range(num_substeps):
                substep_fn(
                    dt=current_dt,
                    flip_ratio=current_flip_ratio,
                    gravity=current_gravity,
                )

        # Update particle colors
        sim.update_default_colors()

        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.4, 0.4, 0.5))

        # Draw particles
        particle_radius = sim.dx * 0.35
        scene.particles(
            sim.pos,
            radius=particle_radius,
            per_vertex_color=sim.color,
        )

        # Draw obstacle if present
        obs_r = sim.obstacle_radius[None]
        if obs_r > 0:
            scene.particles(
                sim.obstacle_pos,
                radius=obs_r * 0.95,
                color=(0.6, 0.3, 0.3),
            )

        canvas.scene(scene)
        window.show()

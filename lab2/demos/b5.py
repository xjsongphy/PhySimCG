"""B5: Surface Reconstruction demo — Marching Cubes from FLIP particles."""
import numpy as np
import taichi as ti
from lab2.core import scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.surface import build_surface_mesh, _init_mc_tables


def run(debug=False):
    ti.init(arch=ti.vulkan)
    _init_mc_tables()

    nx, ny, nz = 32, 48, 32
    scene = "Dam Break with Obstacle"
    num_particles = scene_particle_count(scene, nx)

    sim = FLIPSimulator(nx, ny, nz, num_particles)
    sim.init_dam_break_with_obstacle()
    # Note: obstacle cylinder is baked into cell_type by init_dam_break_with_obstacle
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    window_size = (1600, 900)
    window = ti.ui.Window("Lab2 - B5 Surface Reconstruction", window_size)
    canvas = window.get_canvas()
    canvas.set_background_color((0.15, 0.15, 0.2))
    scene_obj = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 0.9, 2.0)
    camera.lookat(0.5, 0.45, 0.5)

    dt = 0.01
    frame_count = 0
    paused = False

    # Mesh fields (updated every 10 frames)
    mesh_verts = ti.Vector.field(3, dtype=float, shape=(1,))
    mesh_tris = ti.field(dtype=int, shape=(1,))
    has_mesh = False
    mesh_update_interval = 15
    start_mesh_frame = 20

    while window.running:
        if not paused:
            if sim.obstacle_count[None] > 0:
                sim.mark_obstacle_cells()
            for _ in range(2):
                sim.substep(dt=dt, flip_ratio=0.95, gravity=-9.8,
                           num_pressure_iters=50)
            frame_count += 1

        sim.update_default_colors()

        # Build surface mesh periodically
        if frame_count >= start_mesh_frame and frame_count % mesh_update_interval == 0 and not paused:
            verts, tris = build_surface_mesh(
                sim.pos, sim.nx, sim.ny, sim.nz, sim.dx,
                threshold=0.5, mc_res=2,
            )
            if verts is not None and len(verts) > 0:
                if verts.shape[0] != mesh_verts.shape[0]:
                    mesh_verts = ti.Vector.field(3, dtype=float, shape=verts.shape[0])
                if 3 * tris.shape[0] != mesh_tris.shape[0]:
                    mesh_tris = ti.field(dtype=int, shape=3 * tris.shape[0])
                mesh_verts.from_numpy(verts)
                mesh_tris.from_numpy(tris.flatten().astype(np.int32))
                has_mesh = True
            else:
                has_mesh = False

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)

        scene_obj.set_camera(camera)
        scene_obj.point_light(pos=(0.3, 1.8, 1.2), color=(0.9, 0.85, 0.75))
        scene_obj.point_light(pos=(0.7, 1.2, -0.5), color=(0.4, 0.5, 0.7))
        scene_obj.ambient_light((0.35, 0.35, 0.4))

        # Surface mesh (main render)
        if has_mesh and mesh_verts.shape[0] > 0:
            scene_obj.mesh(mesh_verts, mesh_tris,
                          color=(0.25, 0.55, 0.85),
                          show_wireframe=True,
                          two_sided=True)

        # Fluid particles (subtle background dots)
        scene_obj.particles(
            sim.pos,
            radius=sim.dx * 0.08,
            per_vertex_color=sim.color,
        )

        # Obstacle indicator
        obs_count = sim.obstacle_count[None]
        if obs_count > 0:
            for o in range(min(obs_count, 4)):
                obs_r = sim.obstacle_radius[o]
                if obs_r > 0:
                    obs_pos_field = ti.Vector.field(3, dtype=float, shape=(1,))
                    obs_pos_field[0] = sim.obstacle_pos[o]
                    scene_obj.particles(
                        obs_pos_field,
                        radius=obs_r * 0.95,
                        color=(0.8, 0.3, 0.3),
                    )

        canvas.scene(scene_obj)

        # --- GUI ---
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.18, 0.18) as g:
            if g.button("Pause/Resume"):
                paused = not paused
            g.text(f"  Frame: {frame_count}")
            g.text(f"  Particles: {sim.num_particles}")
            if has_mesh:
                g.text(f"  Verts: {mesh_verts.shape[0]}")
                g.text(f"  Tris: {mesh_tris.shape[0] // 3}")
            else:
                g.text(f"  Mesh: building...")

        window.show()


if __name__ == "__main__":
    run()

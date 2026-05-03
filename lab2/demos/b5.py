"""B5: Surface Reconstruction demo — Marching Cubes from FLIP particles."""
import numpy as np
import taichi as ti
from lab2.core import scene_particle_count
from lab2.flip import FLIPSimulator


def run(debug=False):
    ti.init(arch=ti.vulkan)

    nx, ny, nz = 24, 48, 24
    scene = "Dam Break with Obstacle"
    num_particles = scene_particle_count(scene, nx)

    sim = FLIPSimulator(nx, ny, nz, num_particles)
    sim.init_dam_break_with_obstacle()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    window_size = (1600, 900)
    window = ti.ui.Window("Lab2 - B5 Surface Reconstruction", window_size)
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.15))
    scene_obj = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 1.0, 2.2)
    camera.lookat(0.5, 0.4, 0.5)

    # Simulation state
    dt = 0.01
    frame_count = 0
    paused = False

    # Mesh fields (updated every 5 frames)
    mesh_verts = ti.Vector.field(3, dtype=float, shape=(1,))
    mesh_tris = ti.field(dtype=int, shape=(1,))
    has_mesh = False

    while window.running:
        # Simulation
        if not paused:
            for _ in range(2):
                sim.substep(dt=dt, flip_ratio=0.95, gravity=-9.8,
                           num_pressure_iters=40)
            frame_count += 1

        sim.update_default_colors()

        # Rebuild surface mesh every 10 frames (reduce frequency to avoid freezing)
        # Skip first 30 frames to allow particles to settle
        if frame_count > 30 and frame_count % 10 == 0 and not paused:
            from lab2.surface import build_surface_mesh
            verts, tris = build_surface_mesh(
                sim.pos, sim.nx, sim.ny, sim.nz, sim.dx,
                threshold=0.3, mc_res=2,
            )
            if verts is not None and len(verts) > 0:
                # Resize fields if needed
                if verts.shape[0] != mesh_verts.shape[0]:
                    mesh_verts = ti.Vector.field(3, dtype=float, shape=verts.shape[0])
                if tris.shape[0] != mesh_tris.shape[0]:
                    mesh_tris = ti.field(dtype=int, shape=3 * tris.shape[0])
                mesh_verts.from_numpy(verts)
                flat_tris = tris.flatten().astype(np.int32)
                mesh_tris.from_numpy(flat_tris)
                has_mesh = True

        # Camera
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)

        # Render
        scene_obj.set_camera(camera)
        scene_obj.point_light(pos=(0.5, 2.0, 1.5), color=(1.0, 1.0, 1.0))
        scene_obj.ambient_light((0.6, 0.6, 0.7))

        # Fluid particles
        scene_obj.particles(
            sim.pos,
            radius=sim.dx * 0.25,
            per_vertex_color=sim.color,
        )

        # Surface mesh (reconstructed)
        if has_mesh and mesh_verts.shape[0] > 0:
            scene_obj.mesh(mesh_verts, mesh_tris, color=(0.3, 0.6, 1.0),
                          show_wireframe=True)

        canvas.scene(scene_obj)

        # GUI
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.18, 0.15) as g:
            if g.button("Pause/Resume"):
                paused = not paused
            g.text(f"Particles: {sim.num_particles}")
            g.text(f"Mesh: {'ON' if has_mesh else 'building...'}")
            if has_mesh:
                g.text(f"Verts: {mesh_verts.shape[0]}")
                g.text(f"Tris: {mesh_tris.shape[0] // 3}")

        window.show()


if __name__ == "__main__":
    run()

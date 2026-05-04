"""B5: Surface Reconstruction demo — Marching Cubes from FLIP particles."""
import numpy as np
import taichi as ti
from lab2.core import scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.surface import build_surface_mesh, _init_mc_tables


def run(debug=False):
    ti.init(arch=ti.gpu)
    _init_mc_tables()

    nx, ny, nz = 32, 48, 32
    scene = "Dam Break"
    num_particles = scene_particle_count(scene, nx)

    sim = FLIPSimulator(nx, ny, nz, num_particles)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    window_size = (1600, 900)
    window = ti.ui.Window("Lab2 - B5 Surface Reconstruction", window_size)
    canvas = window.get_canvas()
    canvas.set_background_color((0.15, 0.15, 0.2))
    scene_obj = window.get_scene()
    camera = ti.ui.Camera()

    _init_target = np.array([0.5, 0.45, 0.5], dtype=np.float32)
    _init_yaw = -0.3
    _init_pitch = 0.5
    _init_dist = 2.0
    cam_target = _init_target.copy()
    cam_yaw = _init_yaw
    cam_pitch = _init_pitch
    cam_dist = _init_dist

    dt = 0.01
    frame_count = 0
    paused = False

    # Mesh fields
    mc_res = 1
    mc_density = ti.field(dtype=float, shape=(nx * mc_res + 1, ny * mc_res + 1, nz * mc_res + 1))
    mesh_verts = ti.Vector.field(3, dtype=float, shape=(1,))
    mesh_tris = ti.field(dtype=int, shape=(1,))
    has_mesh = False
    mesh_update_interval = 15
    start_mesh_frame = 2

    prev_cx, prev_cy = 0.0, 0.0

    while window.running:
        if not paused:
            for _ in range(2):
                sim.substep(dt=dt, flip_ratio=0.95, gravity=-9.8,
                           num_pressure_iters=30)
            frame_count += 1

        # Build surface mesh periodically
        if frame_count >= start_mesh_frame and frame_count % mesh_update_interval == 0 and not paused:
            verts, tris = build_surface_mesh(
                sim.pos, sim.nx, sim.ny, sim.nz, sim.dx,
                threshold=0.25, mc_res=mc_res, density_ti=mc_density,
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

        # Camera
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        over_gui = cx < 0.25

        dx_mouse = cx - prev_cx
        dy_mouse = cy - prev_cy
        if abs(dx_mouse) < 0.002 and abs(dy_mouse) < 0.002:
            dx_mouse = dy_mouse = 0.0

        cam_pos = np.array([
            cam_target[0] + cam_dist * np.cos(cam_pitch) * np.cos(cam_yaw),
            cam_target[1] + cam_dist * np.sin(cam_pitch),
            cam_target[2] + cam_dist * np.cos(cam_pitch) * np.sin(cam_yaw),
        ], dtype=np.float32)

        forward = cam_target - cam_pos

        if not over_gui:
            if rmb:
                cam_yaw += dx_mouse * 3.0
                cam_pitch -= dy_mouse * 3.0
                cam_pitch = np.clip(cam_pitch, -1.55, 1.55)
            if lmb:
                forward_h = forward.copy()
                forward_h[1] = 0
                forward_h = forward_h / (np.linalg.norm(forward_h) + 1e-8)
                right_pan = np.array([forward_h[2], 0, -forward_h[0]])
                pan_speed = cam_dist * 0.8
                cam_target += right_pan * (dx_mouse * pan_speed) - np.array([0, dy_mouse * pan_speed, 0])

        if window.is_pressed('r'):
            cam_dist = max(0.2, cam_dist - 0.05)
        if window.is_pressed('f'):
            cam_dist = min(10.0, cam_dist + 0.05)

        prev_cx, prev_cy = cx, cy
        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])

        scene_obj.set_camera(camera)
        scene_obj.point_light(pos=(0.3, 1.8, 1.2), color=(1.0, 0.95, 0.85))
        scene_obj.point_light(pos=(0.7, 1.2, -0.5), color=(0.5, 0.6, 0.8))
        scene_obj.point_light(pos=(0.5, 0.8, 1.5), color=(0.3, 0.3, 0.6))
        scene_obj.ambient_light((0.2, 0.22, 0.28))

        # Surface mesh
        if has_mesh and mesh_verts.shape[0] > 0:
            scene_obj.mesh(mesh_verts, mesh_tris,
                          color=(0.3, 0.6, 0.9),
                          show_wireframe=False,
                          two_sided=True)

        canvas.scene(scene_obj)

        # --- GUI ---
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.18, 0.20) as g:
            if g.button("Pause/Resume"):
                paused = not paused
            if g.button("Reset Camera"):
                cam_target[:] = _init_target
                cam_yaw = _init_yaw
                cam_pitch = _init_pitch
                cam_dist = _init_dist
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

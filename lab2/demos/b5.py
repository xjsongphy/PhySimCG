"""B5: Surface Reconstruction demo — Marching Cubes from FLIP particles."""
import os
import re
import shutil
import tempfile
import time
import numpy as np
import taichi as ti
import imageio
from lab2.core import scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.surface import build_surface_mesh


def _save_gif_from_dir(tmp_dir, fps=20):
    """Read PNG frames from temp dir, combine into GIF, then clean up."""
    pngs = sorted(
        [f for f in os.listdir(tmp_dir) if f.endswith(".png")],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    if not pngs:
        print("[record] No frames captured.")
        return
    frames = []
    for p in pngs:
        f = imageio.v3.imread(os.path.join(tmp_dir, p))
        s = f[::2, ::2]
        if s.shape[-1] == 4:
            s = s[..., :3]
        frames.append(s)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(os.path.dirname(__file__), f"record_b5_{ts}.gif")
    imageio.mimsave(out, frames, fps=fps, loop=0)
    shutil.rmtree(tmp_dir)
    print(f"[record] GIF saved: {out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}px)")

def run(debug=False):
    ti.init(arch=ti.gpu)

    nx, ny, nz = 20, 30, 20
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
    _recording = False
    _record_dir = None
    _record_count = 0

    # V-t curve
    sim.save_init_volume()
    _vt_max_pts = 300
    _vt_curve_np = np.zeros((0, 3), dtype=np.float32)
    _vt_curve_field = ti.Vector.field(3, dtype=float, shape=(_vt_max_pts,))
    _vt_frame_idx = 0

    # Container box wireframe
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
    box_pts = _make_box_edges(0.0, 1.0, n=30)
    box_field = ti.Vector.field(3, dtype=float, shape=box_pts.shape[0])
    box_field.from_numpy(box_pts)

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
            # V-t curve update
            _vt_frame_idx += 1
            if _vt_frame_idx % 2 == 0:
                sim.compute_fluid_volume()
                vr = sim._fluid_vol_ratio[0]
                n = min(_vt_curve_np.shape[0], _vt_max_pts - 1)
                px = 0.05 + n / _vt_max_pts * 0.9
                py = 1.05 + max(0.0, min(1.0, (vr - 0.8) / 0.4)) * 0.17
                pt = np.array([[px, py, 0.02]], dtype=np.float32)
                if _vt_curve_np.shape[0] < _vt_max_pts:
                    _vt_curve_np = np.vstack([_vt_curve_np, pt])
                else:
                    _vt_curve_np = np.roll(_vt_curve_np, -1, axis=0)
                    _vt_curve_np[-1] = pt

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

        # Container wireframe
        scene_obj.particles(box_field, radius=sim.dx * 0.05, color=(0.5, 0.5, 0.5))

        # Surface mesh
        if has_mesh and mesh_verts.shape[0] > 0:
            scene_obj.mesh(mesh_verts, mesh_tris,
                          color=(0.3, 0.6, 0.9),
                          show_wireframe=False,
                          two_sided=True)

        # V-t curve
        n_vt = _vt_curve_np.shape[0]
        if n_vt >= 2:
            for i in range(n_vt):
                _vt_curve_field[i] = _vt_curve_np[i]
            for i in range(n_vt, _vt_max_pts):
                _vt_curve_field[i] = _vt_curve_np[-1]
            scene_obj.lines(_vt_curve_field, width=2.0, color=(0.2, 1.0, 0.4))

        canvas.scene(scene_obj)

        # --- GUI ---
        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.18, 0.24) as g:
            dt = g.slider_float("dt", dt, 0.002, 0.02)
            if g.button("Pause/Resume"):
                paused = not paused
            if g.button("Reset Sim"):
                sim.init_dam_break()
                sim.relabel_and_density()
                sim.init_colors()
                sim.store_initial_density()
                frame_count = 0
                has_mesh = False
                mc_density.fill(0.0)
            if g.button("Reset Camera"):
                cam_target[:] = _init_target
                cam_yaw = _init_yaw
                cam_pitch = _init_pitch
                cam_dist = _init_dist
            if g.button("Stop GIF" if _recording else "Record GIF"):
                if not _recording:
                    _recording = True
                    _record_dir = tempfile.mkdtemp(prefix="lab2rec_b5_")
                    _record_count = 0
                else:
                    _recording = False
                    if _record_dir:
                        _save_gif_from_dir(_record_dir)
                    _record_dir = None
                    _record_count = 0
            if _recording:
                g.text(f"  REC {_record_count}")
            sim.compute_fluid_volume()
            g.text(f"  Vol ratio: {sim._fluid_vol_ratio[0]:.3f}")
            g.text(f"  Frame: {frame_count}")
            g.text(f"  Particles: {sim.num_particles}")
            if has_mesh:
                g.text(f"  Verts: {mesh_verts.shape[0]}")
                g.text(f"  Tris: {mesh_tris.shape[0] // 3}")
            else:
                g.text(f"  Mesh: building...")

        if _recording and frame_count % 3 == 0:
            frame_path = os.path.join(_record_dir, f"frame_{_record_count:06d}.png")
            window.save_image(frame_path)
            _record_count += 1

        window.show()


if __name__ == "__main__":
    run()

from __future__ import annotations

import taichi as ti
import numpy as np
import time
import logging
import os
import shutil
import tempfile
import imageio

from lab3.constants import ConstraintMode, FEMConfig, GUIVisibilityConfig, MaterialType
from lab3.core import FEMSystem
from lab3.models import CorotatedModel, NeoHookeanModel, StVKModel
from lab3.solver import BaseFEMSolver, ExplicitFEMSolver, ImplicitNewtonCGSolver


_MATERIAL_NAMES = ["StVK", "NeoHookean", "Corotated"]
_CONSTRAINT_NAMES = ["Top Fixed", "Side Fixed", "Both Sides Fixed", "Top+Bottom Fixed"]
_CAMERA_FOV_DEG = 60.0
_CONSTRAINT_DESC = {
    0: "Top fixed, rest free.",
    1: "Single x-min side fixed, body can bend.",
    2: "Both x sides fixed, middle deforms.",
    3: "Top and bottom fixed, middle layer is freer.",
}


def _rebuild_model_from_ui(solver: BaseFEMSolver, cfg: FEMConfig, material_name: str) -> None:
    if material_name == "StVK":
        cfg.material_type = MaterialType.STVK
        solver.set_material(StVKModel(cfg.youngs_modulus, cfg.poisson_ratio))
    elif material_name == "NeoHookean":
        cfg.material_type = MaterialType.NEO_HOOKEAN
        solver.set_material(NeoHookeanModel(cfg.youngs_modulus, cfg.poisson_ratio))
    else:
        cfg.material_type = MaterialType.COROTATED
        solver.set_material(CorotatedModel(cfg.youngs_modulus, cfg.poisson_ratio))


def _material_index_from_type(material_type: MaterialType) -> int:
    if material_type == MaterialType.STVK:
        return 0
    if material_type == MaterialType.NEO_HOOKEAN:
        return 1
    return 2


def _to_mat4(m) -> np.ndarray:
    arr = np.array(m, dtype=np.float32)
    if arr.size == 16:
        return arr.reshape(4, 4)
    return arr


def _camera_basis(cam_pos: np.ndarray, cam_target: np.ndarray):
    forward = cam_target - cam_pos
    forward_n = forward / (np.linalg.norm(forward) + 1.0e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(forward_n[1]) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(forward_n, world_up)
    right = right / (np.linalg.norm(right) + 1.0e-8)
    up = np.cross(right, forward_n)
    return forward_n, right, up


def _cursor_ray_from_camera(
    cx: float,
    cy: float,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    aspect: float,
    fov_deg: float = _CAMERA_FOV_DEG,
):
    forward_n, right, up = _camera_basis(cam_pos, cam_target)
    fov = np.radians(fov_deg)
    ndc_x = 2.0 * cx - 1.0
    # Taichi cursor coordinates are window-local with origin at lower-left.
    ndc_y = 2.0 * cy - 1.0
    # Reverse of perspective projection: ray in camera basis.
    # dir = forward + x*tan(fov/2)*aspect*right + y*tan(fov/2)*up
    t = np.tan(fov * 0.5)
    ray_dir = forward_n + (ndc_x * aspect * t) * right + (ndc_y * t) * up
    ray_dir /= (np.linalg.norm(ray_dir) + 1.0e-8)
    return cam_pos.astype(np.float32), ray_dir.astype(np.float32)


def _project_to_screen(
    p_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    aspect: float,
    fov_deg: float = _CAMERA_FOV_DEG,
):
    forward_n, right, up = _camera_basis(cam_pos, cam_target)
    to_p = p_world - cam_pos
    depth = float(np.dot(to_p, forward_n))
    if depth <= 1.0e-6:
        return None
    fov = np.radians(fov_deg)
    hh = depth * np.tan(fov * 0.5)
    hw = hh * aspect
    sx = (np.dot(to_p, right) / (hw + 1.0e-8) + 1.0) * 0.5
    # Return to Taichi's lower-left-origin normalized window coordinates.
    sy = (np.dot(to_p, up) / (hh + 1.0e-8) + 1.0) * 0.5
    return float(sx), float(sy), depth


def _cursor_over_gui(cx: float, cy: float) -> bool:
    # sub_window uses top-left origin (+y down), get_cursor_pos uses bottom-left origin (+y up)
    # Transform: y_cursor_bottom = 1 - y_subwindow_top - height
    m = 0.015  # margin for imgui chrome

    # Scene: sub_window(..., 0.02, 0.02, 0.24, 0.26)
    # x: [0.02, 0.26], y_top: 0.02, y_bottom: 0.02+0.26=0.28
    # Transformed y: [1-0.28, 1-0.02] = [0.72, 0.98]
    if 0.02 - m <= cx <= 0.26 + m and 0.72 - m <= cy <= 0.98 + m:
        return True

    # Controls: sub_window(..., 0.02, 0.20, 0.26, 0.56)
    # x: [0.02, 0.28], y_top: 0.20, y_bottom: 0.20+0.56=0.76
    # Transformed y: [1-0.76, 1-0.20] = [0.24, 0.80]
    if 0.02 - m <= cx <= 0.28 + m and 0.24 - m <= cy <= 0.80 + m:
        return True

    # Render: sub_window(..., 0.30, 0.02, 0.18, 0.20)
    # x: [0.30, 0.48], y_top: 0.02, y_bottom: 0.02+0.20=0.22
    # Transformed y: [1-0.22, 1-0.02] = [0.78, 0.98]
    if 0.30 - m <= cx <= 0.48 + m and 0.78 - m <= cy <= 0.98 + m:
        return True

    return False


def _save_gif_from_dir(tmp_dir, fps=15):
    """Read PNG frames from temp dir, combine into GIF, then clean up."""
    import re
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
        s = f[::2, ::2]  # Downsample 2x
        if s.shape[-1] == 4:
            s = s[..., :3]
        frames.append(s)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(os.path.dirname(__file__), f"record_{ts}.gif")
    imageio.mimsave(out, frames, fps=fps, loop=0)
    shutil.rmtree(tmp_dir)
    print(f"[record] GIF saved: {out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}px)")


def run_gui(
    system: FEMSystem,
    solver: BaseFEMSolver,
    cfg: FEMConfig,
    ui_cfg: GUIVisibilityConfig | None = None,
    mesh_presets: dict[str, dict] | None = None,
    rebuild_sim=None,
    logger: logging.Logger | None = None,
    debug: bool = False,
    defaults=None,
) -> None:
    if ui_cfg is None:
        ui_cfg = GUIVisibilityConfig()

    window = ti.ui.Window("Lab3 - FEM", (1400, 900))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    cam_target = np.array([4.0, 1.0, 1.0], dtype=np.float32)
    cam_yaw = -1.1
    cam_pitch = 0.25
    cam_dist = 11.0
    cam_pos = np.array([9.0, 5.0, 12.0], dtype=np.float32)

    paused = False
    material_name = _MATERIAL_NAMES[_material_index_from_type(cfg.material_type)]
    show_particles = True
    show_wireframe = True
    show_lighting = True
    prev_lmb = False
    prev_rmb = False
    prev_cursor_valid = False
    prev_cursor_x = 0.0
    prev_cursor_y = 0.0
    # Interaction state machine: lock on press, reset on release (mutually exclusive)
    # 'none' / 'gui' / 'particle' / 'camera'
    lmb_action = "none"
    # 'none' / 'gui' / 'orbit'
    rmb_action = "none"
    constraint_idx = int(cfg.constraint_mode.value)
    current_mesh_name = next(iter(mesh_presets.keys())) if mesh_presets else ""
    target_fps = 120.0
    frame_dt = 1.0 / target_fps
    last_frame_t = time.perf_counter()
    frame_id = 0

    # Recording state
    _recording = False
    _record_dir = None
    _record_count = 0
    _record_frame = 0

    while window.running:
        frame_begin_t = time.perf_counter()
        now_t = time.perf_counter()
        elapsed = now_t - last_frame_t
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)
        last_frame_t = time.perf_counter()

        # ====== Capture mouse state BEFORE GUI rendering ======
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        in_window = 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        over_gui = _cursor_over_gui(cx, cy)
        ws = window.get_window_shape()
        aspect = float(ws[0]) / max(float(ws[1]), 1.0)
        ray = _cursor_ray_from_camera(cx, cy, cam_pos, cam_target, aspect, fov_deg=_CAMERA_FOV_DEG)

        # --- LMB: determine type on press, lock for entire drag ---
        if lmb and not prev_lmb:
            if over_gui or not in_window:
                lmb_action = "gui"
            elif ray is not None:
                system.begin_drag(ray[0], ray[1])
                if int(system.drag_vertex_idx[None]) >= 0:
                    lmb_action = "particle"
                else:
                    x_np = system.x.to_numpy()
                    best_i = -1
                    best_dist = 1e9
                    for i in range(x_np.shape[0]):
                        proj = _project_to_screen(x_np[i], cam_pos, cam_target, aspect, fov_deg=_CAMERA_FOV_DEG)
                        if proj is None:
                            continue
                        sx, sy, depth = proj
                        d = np.hypot(sx - cx, sy - cy)
                        if d < max(0.02, 0.12 / max(depth, 1e-6)) and d < best_dist:
                            best_dist = d
                            best_i = i
                    if best_i >= 0 and hasattr(system, "begin_drag_vertex"):
                        system.begin_drag_vertex(best_i, ray[0], ray[1])
                        if int(system.drag_vertex_idx[None]) >= 0:
                            lmb_action = "particle"
                        else:
                            lmb_action = "camera"
                    else:
                        lmb_action = "camera"
            else:
                lmb_action = "camera"

        if lmb_action == "particle" and lmb and ray is not None:
            system.drag_to(ray[0], ray[1])
        if not lmb and prev_lmb:
            if lmb_action == "particle":
                system.end_drag()
            lmb_action = "none"

        # --- RMB: determine type on press ---
        if rmb and not prev_rmb:
            if over_gui or not in_window:
                rmb_action = "gui"
            else:
                rmb_action = "orbit"
        if not rmb and prev_rmb:
            rmb_action = "none"

        # Camera controls
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y
            if rmb_action == "orbit":
                cam_yaw += dx_mouse * 3.0
                cam_pitch -= dy_mouse * 3.0
                cam_pitch = np.clip(cam_pitch, -1.55, 1.55)
            elif lmb_action == "camera":
                forward = cam_target - cam_pos
                forward[1] = 0.0
                forward /= (np.linalg.norm(forward) + 1e-8)
                right_pan = np.array([forward[2], 0.0, -forward[0]], dtype=np.float32)
                pan_speed = max(0.2, cam_dist * 0.8)
                cam_target += right_pan * (dx_mouse * pan_speed)
                cam_target -= np.array([0.0, dy_mouse * pan_speed, 0.0], dtype=np.float32)

        if window.is_pressed("r"):
            cam_dist = max(0.5, cam_dist - 0.08)
        if window.is_pressed("f"):
            cam_dist = min(40.0, cam_dist + 0.08)

        cam_pos[0] = cam_target[0] + cam_dist * np.cos(cam_pitch) * np.cos(cam_yaw)
        cam_pos[1] = cam_target[1] + cam_dist * np.sin(cam_pitch)
        cam_pos[2] = cam_target[2] + cam_dist * np.cos(cam_pitch) * np.sin(cam_yaw)
        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        camera.fov(_CAMERA_FOV_DEG)

        prev_lmb = lmb
        prev_rmb = rmb
        prev_cursor_x, prev_cursor_y = cx, cy
        prev_cursor_valid = True
        # ====== Mouse state capture complete ======

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                paused = not paused
                if logger is not None:
                    logger.info("Pause toggled -> %s", paused)

        gui = window.get_gui()

        # --- Scene panel ---
        with gui.sub_window("Scene", 0.02, 0.02, 0.24, 0.26) as g:
            g.text("=== Scene ===")
            # Debug: show mouse position and state
            g.text(f"pos: ({cx:.2f},{cy:.2f})")
            g.text(f"over_gui:{over_gui} action:{lmb_action}")
            g.text("---")
            g.text("RMB + drag: orbit")
            g.text("LMB hit point: drag force")
            g.text("LMB empty: camera pan")
            g.text("R / F: zoom")
            g.text("SPACE: pause/resume")
            if g.button("Reset Sim"):
                system.reset_state()
            if g.button("Stop GIF" if _recording else "Record GIF"):
                if not _recording:
                    _recording = True
                    _record_dir = tempfile.mkdtemp(prefix="lab3rec_")
                    _record_count = 0
                    _record_frame = 0
                else:
                    _recording = False
                    if _record_dir:
                        _save_gif_from_dir(_record_dir)
                    _record_dir = None
                    _record_count = 0
            if _recording:
                g.text(f"  REC {_record_count}")
            g.text("Constraint:")
            new_constraint_idx = constraint_idx
            for i, name in enumerate(_CONSTRAINT_NAMES):
                if g.button(name):
                    new_constraint_idx = i
            if new_constraint_idx != constraint_idx:
                constraint_idx = int(new_constraint_idx)
                system.set_constraint_mode(ConstraintMode(constraint_idx))
                system.reset_state()
            g.text(_CONSTRAINT_DESC.get(constraint_idx, ""))
            if mesh_presets and rebuild_sim is not None:
                g.text("Mesh:")
                g.text(f"Current: {current_mesh_name}")
                for name in mesh_presets.keys():
                    if g.button(name):
                        current_mesh_name = name
                        if logger is not None:
                            logger.info("Rebuild sim mesh=%s", name)
                        system, solver = rebuild_sim(name)
                        constraint_idx = int(cfg.constraint_mode.value)

        # --- Controls panel ---
        p = ui_cfg.parameters
        if p.show_panel:
            with gui.sub_window("Controls", 0.02, 0.20, 0.26, 0.56) as panel:
                panel.text("=== Simulation ===")

                if p.show_paused:
                    paused = panel.checkbox("Paused", paused)

                if p.show_implicit_toggle:
                    implicit_mode = panel.checkbox("Implicit (Newton+CG)", cfg.use_implicit)
                    if implicit_mode != cfg.use_implicit:
                        cfg.use_implicit = implicit_mode
                        solver = ImplicitNewtonCGSolver(system, cfg) if cfg.use_implicit else ExplicitFEMSolver(system, cfg)

                # --- Time Integration ---
                panel.text("--- Time ---")
                if p.show_dt:
                    cfg.dt = panel.slider_float("dt", cfg.dt, 1.0e-2, 5.0e-2)
                    # Show stability hint
                    if not cfg.use_implicit:
                        # Estimate stability limit: dt_critical ≈ h / sqrt(E/rho)
                        wave_speed = np.sqrt(cfg.youngs_modulus / cfg.density)
                        cell_size = cfg.cell_size if hasattr(cfg, 'cell_size') else 0.5
                        dt_critical = cell_size / wave_speed
                        # Drag force significantly reduces stable dt
                        dt_safe = dt_critical / 10.0  # Conservative estimate
                        effective_dt = cfg.dt / max(1, cfg.substeps)
                        panel.text(f"dt per step: {effective_dt:.1e}s")
                        if effective_dt > dt_safe:
                            panel.text(f"[WARNING] dt too high!", color=(1.0, 0.2, 0.2))
                            panel.text(f"Stability limit: ~{dt_safe:.1e}s", color=(1.0, 0.2, 0.2))
                        elif effective_dt > dt_safe * 0.5:
                            panel.text(f"[CAUTION] Near stability limit", color=(1.0, 0.67, 0.0))
                            panel.text(f"Safe limit: ~{dt_safe:.1e}s", color=(1.0, 0.67, 0.0))
                        else:
                            panel.text(f"Stability limit: ~{dt_safe:.1e}s", color=(0.2, 1.0, 0.2))
                if p.show_substeps:
                    cfg.substeps = panel.slider_int("substeps", cfg.substeps, 1, 12)

                # --- Material Properties ---
                panel.text("--- Material ---")
                if p.show_damping:
                    cfg.damping = panel.slider_float("damping", cfg.damping, 0.90, 1.0)
                if p.show_youngs:
                    cfg.youngs_modulus = panel.slider_float("Young's", cfg.youngs_modulus, 1.0e3, 8.0e4)
                if p.show_poisson:
                    cfg.poisson_ratio = panel.slider_float("Poisson", cfg.poisson_ratio, 0.05, 0.45)

                # --- Drag Force (Interaction) ---
                panel.text("--- Drag Force ---")
                if hasattr(system, "drag_stiffness"):
                    drag_k = system.drag_stiffness
                    drag_c = system.drag_damping
                    new_drag_k = panel.slider_float("drag_k", drag_k, 100.0, 5000.0)
                    new_drag_c = panel.slider_float("drag_damping", drag_c, 1.0, 50.0)
                    if abs(new_drag_k - drag_k) > 0.1 or abs(new_drag_c - drag_c) > 0.1:
                        system.set_drag_params(new_drag_k, new_drag_c)

                # --- Environment ---
                panel.text("--- Environment ---")
                if p.show_gravity_y:
                    gy = panel.slider_float("gravity_y", cfg.gravity[1], -9.8, 0.0)
                    cfg.gravity = (cfg.gravity[0], gy, cfg.gravity[2])

                # --- Boundary Vibration (B1 bonus) ---
                if p.show_boundary_vibration:
                    panel.text("--- Boundary Vibration ---")
                    cfg.enable_boundary_vibration = panel.checkbox("Vibrate fixed", cfg.enable_boundary_vibration)
                    if cfg.enable_boundary_vibration:
                        cfg.boundary_vibration_amplitude = panel.slider_float("Amplitude", cfg.boundary_vibration_amplitude, 0.0, 0.5)
                        cfg.boundary_vibration_frequency = panel.slider_float("Frequency", cfg.boundary_vibration_frequency, 0.5, 10.0)

                # --- Implicit Solver ---
                panel.text("--- Implicit Solver ---")
                if p.show_newton_iters:
                    cfg.newton_max_iters = panel.slider_int("newton_iters", cfg.newton_max_iters, 2, 30)
                if p.show_cg_iters:
                    cfg.cg_max_iters = panel.slider_int("cg_iters", cfg.cg_max_iters, 10, 150)

                if p.show_material_dropdown:
                    current_idx = _material_index_from_type(cfg.material_type)
                    selected_idx = current_idx
                    panel.text("Material:")
                    for i, name in enumerate(_MATERIAL_NAMES):
                        if panel.button(name):
                            selected_idx = i
                    selected_name = _MATERIAL_NAMES[selected_idx]
                    if selected_name != material_name:
                        material_name = selected_name
                        _rebuild_model_from_ui(solver, cfg, material_name)
                        system.reset_state()

                if p.show_material_text:
                    panel.text(f"Material: {material_name}")

                # Reset defaults button
                if defaults is not None:
                    if panel.button("Reset Defaults"):
                        # Restore default parameters
                        cfg.density = defaults.density
                        cfg.youngs_modulus = defaults.youngs_modulus
                        cfg.poisson_ratio = defaults.poisson_ratio
                        cfg.damping = defaults.damping
                        cfg.dt = defaults.dt
                        cfg.substeps = defaults.substeps
                        cfg.gravity = defaults.gravity
                        # Restore material parameters
                        solver.sync_material_from_config()
                        # Reset simulation
                        system.reset_state()
                        if logger is not None:
                            logger.info("Reset to defaults")

        # --- Render panel ---
        r = ui_cfg.render
        if r.show_panel:
            with gui.sub_window("Render", 0.30, 0.02, 0.18, 0.20) as g:
                g.text("=== Render ===")
                if r.show_particles:
                    show_particles = g.checkbox("Particles", show_particles)
                if r.show_wireframe:
                    show_wireframe = g.checkbox("Wireframe", show_wireframe)
                if r.show_lighting:
                    show_lighting = g.checkbox("Lighting", show_lighting)

        solver_dt = 0.0
        if not paused:
            step_t0 = time.perf_counter()
            solver.step()
            solver_dt = time.perf_counter() - step_t0
        scene.set_camera(camera)
        if show_lighting:
            scene.point_light(pos=(8, 12, 10), color=(1.0, 1.0, 1.0))
            scene.ambient_light((0.3, 0.3, 0.3))
        else:
            scene.ambient_light((1.0, 1.0, 1.0))
        if show_particles:
            scene.particles(system.x, radius=0.05, color=(0.2, 0.7, 1.0))
        if show_wireframe:
            system.build_line_points()
            scene.lines(system.line_points, width=1.0, color=(0.85, 0.85, 0.9))

        canvas.scene(scene)
        # Capture frame for GIF recording
        if _recording and _record_frame % 3 == 0:
            frame_path = os.path.join(_record_dir, f"frame_{_record_count:06d}.png")
            window.save_image(frame_path)
            _record_count += 1
        _record_frame += 1
        render_dt = time.perf_counter() - frame_begin_t - solver_dt
        window.show()
        frame_id += 1
        if debug and logger is not None and frame_id % 30 == 0:
            x_np = system.x.to_numpy()
            v_np = system.v.to_numpy()
            logger.debug(
                "frame=%d paused=%s dt=%.6f substeps=%d verts=%d y_min=%.4f y_max=%.4f v_mean=%.5f solver=%.4fs render=%.4fs",
                frame_id,
                paused,
                cfg.dt,
                cfg.substeps,
                int(system.num_vertices),
                float(x_np[:, 1].min()),
                float(x_np[:, 1].max()),
                float(np.linalg.norm(v_np, axis=1).mean()),
                solver_dt,
                max(0.0, render_dt),
            )

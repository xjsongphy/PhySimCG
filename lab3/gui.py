from __future__ import annotations

import taichi as ti
import numpy as np
import time

from lab3.constants import ConstraintMode, FEMConfig, GUIVisibilityConfig, MaterialType
from lab3.core import FEMSystem
from lab3.models import CorotatedModel, NeoHookeanModel, StVKModel
from lab3.solver import BaseFEMSolver, ExplicitFEMSolver, ImplicitNewtonCGSolver


_MATERIAL_NAMES = ["StVK", "NeoHookean", "Corotated"]
_CONSTRAINT_NAMES = ["Top Fixed", "Side Fixed", "Both Sides Fixed", "Top+Bottom Fixed"]
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


def _cursor_ray(window: ti.ui.Window, camera: ti.ui.Camera, aspect: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        cx, cy = window.get_cursor_pos()
        x_ndc = 2.0 * cx - 1.0
        y_ndc = 1.0 - 2.0 * cy
        # Point on the image plane (near clip) in NDC.
        clip_near = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float32)

        proj = _to_mat4(camera.get_projection_matrix(aspect))
        view = _to_mat4(camera.get_view_matrix())
        inv_vp = np.linalg.inv(proj @ view)
        near_h = inv_vp @ clip_near
        if abs(near_h[3]) < 1.0e-8:
            return None
        near_world = near_h[:3] / near_h[3]

        inv_view = np.linalg.inv(view)
        eye_h = inv_view @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if abs(eye_h[3]) < 1.0e-8:
            return None
        origin = eye_h[:3] / eye_h[3]
        direction = near_world - origin
        n = np.linalg.norm(direction)
        if n < 1.0e-8:
            return None
        direction /= n
        return origin.astype(np.float32), direction.astype(np.float32)
    except Exception:
        return None


def run_gui(system: FEMSystem, solver: BaseFEMSolver, cfg: FEMConfig, ui_cfg: GUIVisibilityConfig | None = None) -> None:
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
    dragging_point = False
    constraint_idx = int(cfg.constraint_mode.value)
    target_fps = 120.0
    frame_dt = 1.0 / target_fps
    last_frame_t = time.perf_counter()

    while window.running:
        now_t = time.perf_counter()
        elapsed = now_t - last_frame_t
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)
        last_frame_t = time.perf_counter()

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                paused = not paused

        gui = window.get_gui()

        # --- Scene panel ---
        with gui.sub_window("Scene", 0.02, 0.02, 0.24, 0.26) as g:
            g.text("=== Scene ===")
            g.text("RMB + drag: camera")
            g.text("LMB hit point: drag force")
            g.text("LMB empty: camera pan")
            g.text("SPACE: pause/resume")
            if g.button("Reset Sim"):
                system.reset_state()
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

                if p.show_dt:
                    cfg.dt = panel.slider_float("dt", cfg.dt, 1.0e-4, 5.0e-3)
                if p.show_substeps:
                    cfg.substeps = panel.slider_int("substeps", cfg.substeps, 1, 20)
                if p.show_damping:
                    cfg.damping = panel.slider_float("damping", cfg.damping, 0.90, 1.0)
                if p.show_youngs:
                    cfg.youngs_modulus = panel.slider_float("Young's", cfg.youngs_modulus, 1.0e3, 8.0e4)
                if p.show_poisson:
                    cfg.poisson_ratio = panel.slider_float("Poisson", cfg.poisson_ratio, 0.05, 0.45)
                if p.show_gravity_y:
                    gy = panel.slider_float("gravity_y", cfg.gravity[1], -9.8, 0.0)
                    cfg.gravity = (cfg.gravity[0], gy, cfg.gravity[2])
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

                if p.show_material_text:
                    panel.text(f"Material: {material_name}")

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

        # Interaction: pick vertex by ray-sphere test, drag to apply spring force.
        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        over_gui = (cx < 0.50)
        aspect = 1400.0 / 900.0
        ray = _cursor_ray(window, camera, aspect)

        # LMB: pick/drag point if hit, otherwise pan camera.
        if lmb and not prev_lmb and not over_gui and ray is not None:
            system.begin_drag(ray[0], ray[1])
            dragging_point = int(system.drag_vertex_idx[None]) >= 0
        if lmb and dragging_point and ray is not None:
            system.drag_to(ray[0], ray[1])
        if (not lmb) and prev_lmb:
            system.end_drag()
            dragging_point = False

        # Camera controls (lab2-style separation)
        if prev_cursor_valid:
            dx_mouse = cx - prev_cursor_x
            dy_mouse = cy - prev_cursor_y
            if not over_gui:
                if rmb:
                    cam_yaw += dx_mouse * 3.0
                    cam_pitch -= dy_mouse * 3.0
                    cam_pitch = np.clip(cam_pitch, -1.55, 1.55)
                elif lmb and not dragging_point:
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

        prev_lmb = lmb
        prev_rmb = rmb
        prev_cursor_x, prev_cursor_y = cx, cy
        prev_cursor_valid = True

        if not paused:
            solver.step()
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
        window.show()

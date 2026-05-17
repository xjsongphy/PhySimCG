from __future__ import annotations

import taichi as ti
import numpy as np

from lab3.constants import FEMConfig, GUIVisibilityConfig, MaterialType
from lab3.core import FEMSystem
from lab3.models import CorotatedModel, NeoHookeanModel, StVKModel
from lab3.solver import BaseFEMSolver, ExplicitFEMSolver, ImplicitNewtonCGSolver


_MATERIAL_NAMES = ["StVK", "NeoHookean", "Corotated"]


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
    camera.position(9.0, 5.0, 12.0)
    camera.lookat(4.0, 1.0, 1.0)
    camera.up(0.0, 1.0, 0.0)

    paused = False
    material_name = _MATERIAL_NAMES[_material_index_from_type(cfg.material_type)]
    show_particles = True
    show_wireframe = True
    show_lighting = True
    prev_lmb = False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                paused = not paused

        gui = window.get_gui()

        # --- Scene panel ---
        with gui.sub_window("Scene", 0.02, 0.02, 0.22, 0.16) as g:
            g.text("=== Scene ===")
            g.text("RMB + drag: camera")
            g.text("SPACE: pause/resume")

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
                    if hasattr(panel, "dropdown"):
                        selected_idx = panel.dropdown("Material", current_idx, _MATERIAL_NAMES)
                    elif hasattr(panel, "combo"):
                        selected_idx = panel.combo("Material", current_idx, _MATERIAL_NAMES)
                    else:
                        selected_idx = panel.slider_int("Material", current_idx, 0, len(_MATERIAL_NAMES) - 1)
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
        lmb = window.is_pressed(ti.ui.LMB)
        aspect = 1400.0 / 900.0
        ray = _cursor_ray(window, camera, aspect)
        if lmb and not prev_lmb and ray is not None:
            system.begin_drag(ray[0], ray[1])
        if lmb and ray is not None:
            system.drag_to(ray[0], ray[1])
        if (not lmb) and prev_lmb:
            system.end_drag()
        prev_lmb = lmb

        if not paused:
            solver.step()

        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        if show_lighting:
            scene.point_light(pos=(8, 12, 10), color=(1.0, 1.0, 1.0))
            scene.ambient_light((0.3, 0.3, 0.3))
        else:
            scene.ambient_light((1.0, 1.0, 1.0))
        if show_particles:
            scene.particles(system.x, radius=0.05, color=(0.2, 0.7, 1.0))
        if show_wireframe:
            scene.lines(system.x, width=1.0, indices=system.edge_indices, color=(0.85, 0.85, 0.9))

        canvas.scene(scene)
        window.show()

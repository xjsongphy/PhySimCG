from __future__ import annotations

import taichi as ti
import numpy as np
import time
import logging
import os
import shutil
import tempfile
import imageio
from typing import Callable, Mapping

from lab3.constants import ConstraintMode, FEMConfig, GUIVisibilityConfig, MaterialType
from lab3.core import FEMSystem
from lab3.models import CorotatedModel, NeoHookeanModel, StVKModel
from lab3.solver import BaseFEMSolver, ExplicitFEMSolver, ImplicitNewtonCGSolver


AnalysisEnergyBreakdown = Mapping[str, float] | float
AnalysisEnergyFn = Callable[[FEMSystem, FEMConfig, BaseFEMSolver], AnalysisEnergyBreakdown]


_MATERIAL_NAMES = ["StVK", "NeoHookean", "Corotated"]
_CONSTRAINT_NAMES = ["Top Fixed", "Side Fixed", "Both Sides Fixed", "Top+Bottom Fixed", "Single Corner Fixed"]
_CAMERA_FOV_DEG = 60.0
_CONSTRAINT_DESC = {
    0: "Top fixed, rest free.",
    1: "Single x-min side fixed, body can bend.",
    2: "Both x sides fixed, middle deforms.",
    3: "Top and bottom fixed, middle layer is freer.",
    4: "Single corner fixed.",
    5: "Two top corners fixed with inward anchor offset.",
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

    # Controls: sub_window(..., 0.72, 0.02, 0.26, 0.56)
    # x: [0.72, 0.98], y_top: 0.20, y_bottom: 0.20+0.56=0.76
    # Transformed y: [1-0.58, 1-0.02] = [0.42, 0.98]
    if 0.72 - m <= cx <= 0.98 + m and 0.42 - m <= cy <= 0.98 + m:
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


def _collision_world_enabled(cw) -> bool:
    return any(col.enabled for col in cw.planes) or any(col.enabled for col in cw.spheres) or any(col.enabled for col in cw.aabbs)


def _sync_collision_enabled(cfg: FEMConfig, system: FEMSystem) -> None:
    if hasattr(system, "collision_world") and system.collision_world is not None:
        cfg.enable_collision = _collision_world_enabled(system.collision_world)


def _draw_plane_grid(scene, normal: np.ndarray, offset: float, color=(0.5, 0.5, 0.5)) -> None:
    pts = []
    if abs(normal[1]) > 0.9:
        y = offset
        for i in range(-10, 11):
            pts.append([-10.0 + i, y, -10.0])
            pts.append([-10.0 + i, y, 10.0])
            pts.append([-10.0, y, -10.0 + i])
            pts.append([10.0, y, -10.0 + i])
    elif abs(normal[0]) > 0.9:
        x = offset
        for i in range(-10, 11):
            pts.append([x, -10.0 + i, -10.0])
            pts.append([x, -10.0 + i, 10.0])
            pts.append([x, -10.0, -10.0 + i])
            pts.append([x, 10.0, -10.0 + i])
    if pts:
        scene.lines(np.asarray(pts, dtype=np.float32), width=2.0, color=color)


def _draw_sphere_wire(scene, center: np.ndarray, radius: float, color=(1.0, 0.4, 0.4)) -> None:
    c = np.asarray(center, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=True)
    rings = [
        np.column_stack((np.full_like(phi, c[0]), c[1] + radius * np.cos(phi), c[2] + radius * np.sin(phi))),
        np.column_stack((c[0] + radius * np.cos(phi), np.full_like(phi, c[1]), c[2] + radius * np.sin(phi))),
        np.column_stack((c[0] + radius * np.cos(phi), c[1] + radius * np.sin(phi), np.full_like(phi, c[2]))),
    ]
    for ring in rings:
        scene.lines(ring.astype(np.float32), width=2.0, color=color)


def _draw_aabb_wire(scene, bmin: np.ndarray, bmax: np.ndarray, color=(1.0, 0.6, 0.2)) -> None:
    bmin = np.asarray(bmin, dtype=np.float32)
    bmax = np.asarray(bmax, dtype=np.float32)
    pts = np.asarray(
        [
            [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]], [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmin[1], bmax[2]], [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]], [bmin[0], bmax[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]], [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmin[2]], [bmin[0], bmax[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]], [bmax[0], bmax[1], bmax[2]],
            [bmin[0], bmin[1], bmax[2]], [bmin[0], bmax[1], bmax[2]],
        ],
        dtype=np.float32,
    )
    scene.lines(pts, width=3.0, color=color)


def _draw_collision_controls(panel, cfg: FEMConfig, system: FEMSystem, logger: logging.Logger | None) -> None:
    if not hasattr(system, "collision_world") or system.collision_world is None:
        return

    cw = system.collision_world
    panel.text("--- Collision Objects ---")
    changed = False
    if len(cw.planes) > 0:
        enabled = panel.checkbox("Ground Plane", bool(cw.planes[0].enabled))
        if enabled != cw.planes[0].enabled:
            cw.planes[0].enabled = enabled
            changed = True
    if len(cw.planes) > 1:
        enabled = panel.checkbox("Wall Plane", bool(cw.planes[1].enabled))
        if enabled != cw.planes[1].enabled:
            cw.planes[1].enabled = enabled
            changed = True
    if len(cw.spheres) > 0:
        enabled = panel.checkbox("Sphere", bool(cw.spheres[0].enabled))
        if enabled != cw.spheres[0].enabled:
            cw.spheres[0].enabled = enabled
            changed = True
    if len(cw.aabbs) > 0:
        enabled = panel.checkbox("Box", bool(cw.aabbs[0].enabled))
        if enabled != cw.aabbs[0].enabled:
            cw.aabbs[0].enabled = enabled
            changed = True

    _sync_collision_enabled(cfg, system)
    panel.text(f"Collision: {'on' if cfg.enable_collision else 'off'}")
    if changed and logger is not None:
        logger.info("Collision toggled -> %s", cfg.enable_collision)


def _stvk_energy_density(F: np.ndarray, mu: float, lmbda: float) -> float:
    dim = F.shape[1]
    E = 0.5 * (F.T @ F - np.eye(dim, dtype=np.float32))
    tr_e = float(np.trace(E))
    return float(mu * np.sum(E * E) + 0.5 * lmbda * tr_e * tr_e)


def _neo_hookean_energy_density(F: np.ndarray, mu: float, lmbda: float) -> float:
    J = max(float(np.linalg.det(F)), 1.0e-8)
    log_j = np.log(J)
    return float(0.5 * mu * (np.sum(F * F) - 3.0) - mu * log_j + 0.5 * lmbda * log_j * log_j)


def _corotated_energy_density(F: np.ndarray, mu: float, lmbda: float) -> float:
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    J = float(np.linalg.det(F))
    return float(mu * np.sum((F - R) * (F - R)) + 0.5 * lmbda * (J - 1.0) * (J - 1.0))


def compute_model_energy_breakdown(system: FEMSystem, cfg: FEMConfig, solver: BaseFEMSolver) -> dict[str, float]:
    x = system.x.to_numpy()
    v = system.v.to_numpy()
    mass = system.mass.to_numpy()
    kinetic = 0.5 * float(np.sum(mass[:, None] * v * v))
    gravity = np.asarray(cfg.gravity, dtype=np.float32)
    gravitational = -float(np.sum(mass * (x @ gravity)))

    mu = float(solver.model.mu)
    lmbda = float(solver.model.lmbda)
    elastic = 0.0
    if hasattr(system, "tets"):
        tets = system.tets.to_numpy()
        dm_inv = system.dm_inv.to_numpy()
        rest_volume = system.rest_volume.to_numpy()
        for e, tet in enumerate(tets):
            x0, x1, x2, x3 = x[tet[0]], x[tet[1]], x[tet[2]], x[tet[3]]
            Ds = np.column_stack((x1 - x0, x2 - x0, x3 - x0)).astype(np.float32)
            F = Ds @ dm_inv[e]
            if cfg.material_type == MaterialType.STVK:
                psi = _stvk_energy_density(F, mu, lmbda)
            elif cfg.material_type == MaterialType.NEO_HOOKEAN:
                psi = _neo_hookean_energy_density(F, mu, lmbda)
            else:
                psi = _corotated_energy_density(F, mu, lmbda)
            elastic += float(rest_volume[e]) * psi
    elif hasattr(system, "tris"):
        tris = system.tris.to_numpy()
        dm_inv = system.dm_inv.to_numpy()
        rest_area = system.rest_area.to_numpy()
        for e, tri in enumerate(tris):
            x0, x1, x2 = x[tri[0]], x[tri[1]], x[tri[2]]
            Ds = np.column_stack((x1 - x0, x2 - x0)).astype(np.float32)
            F = Ds @ dm_inv[e]
            elastic += float(rest_area[e]) * _stvk_energy_density(F, mu, lmbda)

    potential = gravitational + elastic
    total = kinetic + potential
    return {
        "kinetic": kinetic,
        "potential": potential,
        "potential_gravity": gravitational,
        "potential_elastic": elastic,
        "total": total,
    }


def compute_model_total_energy(system: FEMSystem, cfg: FEMConfig, solver: BaseFEMSolver) -> float:
    return compute_model_energy_breakdown(system, cfg, solver)["total"]


_compute_total_energy = compute_model_total_energy


def _mesh_density_label(system: FEMSystem, cfg: FEMConfig, mesh_name: str) -> str:
    if hasattr(system, "_nx") and hasattr(system, "_ny"):
        return f"{mesh_name}:nx={int(system._nx)},ny={int(system._ny)},verts={int(system.num_vertices)}"
    return (
        f"{mesh_name}:wx={int(cfg.wx)},wy={int(cfg.wy)},wz={int(cfg.wz)},"
        f"cell={float(cfg.cell_size):.6g},verts={int(system.num_vertices)}"
    )


def _collision_summary(system: FEMSystem) -> str:
    if not hasattr(system, "collision_world") or system.collision_world is None:
        return "none"
    cw = system.collision_world
    enabled = []
    for i, col in enumerate(cw.planes):
        if col.enabled:
            enabled.append(f"plane{i}")
    for i, col in enumerate(cw.spheres):
        if col.enabled:
            enabled.append(f"sphere{i}")
    for i, col in enumerate(cw.aabbs):
        if col.enabled:
            enabled.append(f"aabb{i}")
    return ",".join(enabled) if enabled else "all_off"


def _system_element_count(system: FEMSystem) -> int:
    if hasattr(system, "num_tets"):
        return int(system.num_tets)
    if hasattr(system, "num_tris"):
        return int(system.num_tris)
    return 0


def _log_sim_config(
    logger: logging.Logger | None,
    event: str,
    demo_name: str,
    scene_style: str,
    mesh_name: str,
    scenario: str,
    system: FEMSystem,
    cfg: FEMConfig,
    solver: BaseFEMSolver,
    analysis_mode: bool,
) -> None:
    if logger is None:
        return
    solver_name = "implicit" if cfg.use_implicit else "explicit"
    logger.info(
        "CONFIG event=%s demo=%s scene=%s scenario=%s analysis=%s solver=%s mesh=%s elements=%d "
        "constraint=%s dt=%.8g substeps=%d material=%s density=%.8g youngs=%.8g poisson=%.8g damping=%.8g "
        "gravity=(%.8g,%.8g,%.8g) newton_iters=%d cg_iters=%d boundary_vibration=%s "
        "boundary_amp=%.8g boundary_freq=%.8g side_stretch=%s side_amp=%.8g side_freq=%.8g "
        "collision=%s collision_objects=%s collision_k=%.8g collision_c=%.8g collision_radius=%.8g",
        event,
        demo_name,
        scene_style,
        scenario,
        analysis_mode,
        solver_name,
        _mesh_density_label(system, cfg, mesh_name),
        _system_element_count(system),
        cfg.constraint_mode.name,
        float(cfg.dt),
        int(cfg.substeps),
        cfg.material_type.name,
        float(cfg.density),
        float(cfg.youngs_modulus),
        float(cfg.poisson_ratio),
        float(cfg.damping),
        float(cfg.gravity[0]),
        float(cfg.gravity[1]),
        float(cfg.gravity[2]),
        int(cfg.newton_max_iters),
        int(cfg.cg_max_iters),
        bool(cfg.enable_boundary_vibration),
        float(cfg.boundary_vibration_amplitude),
        float(cfg.boundary_vibration_frequency),
        bool(cfg.enable_side_stretch),
        float(cfg.side_stretch_amplitude),
        float(cfg.side_stretch_frequency),
        bool(cfg.enable_collision),
        _collision_summary(system),
        float(cfg.collision_k),
        float(cfg.collision_c),
        float(cfg.collision_particle_radius),
    )


def _log_analysis_sample(
    logger: logging.Logger | None,
    demo_name: str,
    scene_style: str,
    mesh_name: str,
    scenario: str,
    system: FEMSystem,
    cfg: FEMConfig,
    solver: BaseFEMSolver,
    energy_fn: AnalysisEnergyFn,
) -> None:
    if logger is None:
        return
    energy = energy_fn(system, cfg, solver)
    if isinstance(energy, Mapping):
        kinetic_energy = float(energy.get("kinetic", 0.0))
        potential_gravity = float(energy.get("potential_gravity", 0.0))
        potential_elastic = float(energy.get("potential_elastic", 0.0))
        potential_energy = float(energy.get("potential", potential_gravity + potential_elastic))
        total_energy = float(energy.get("total", kinetic_energy + potential_energy))
    else:
        kinetic_energy = float("nan")
        potential_gravity = float("nan")
        potential_elastic = float("nan")
        potential_energy = float("nan")
        total_energy = float(energy)
    solver_name = "implicit" if cfg.use_implicit else "explicit"
    logger.info(
        "ANALYSIS demo=%s scene=%s scenario=%s t=%.8f solver=%s mesh=%s elements=%d constraint=%s "
        "dt=%.8g substeps=%d material=%s density=%.8g youngs=%.8g poisson=%.8g damping=%.8g "
        "gravity=(%.8g,%.8g,%.8g) collision=%s collision_objects=%s kinetic_energy=%.12g "
        "potential_energy=%.12g potential_gravity=%.12g potential_elastic=%.12g total_energy=%.12g",
        demo_name,
        scene_style,
        scenario,
        float(system.get_sim_time()) if hasattr(system, "get_sim_time") else 0.0,
        solver_name,
        _mesh_density_label(system, cfg, mesh_name),
        _system_element_count(system),
        cfg.constraint_mode.name,
        float(cfg.dt),
        int(cfg.substeps),
        cfg.material_type.name,
        float(cfg.density),
        float(cfg.youngs_modulus),
        float(cfg.poisson_ratio),
        float(cfg.damping),
        float(cfg.gravity[0]),
        float(cfg.gravity[1]),
        float(cfg.gravity[2]),
        bool(cfg.enable_collision),
        _collision_summary(system),
        kinetic_energy,
        potential_energy,
        potential_gravity,
        potential_elastic,
        total_energy,
    )


def run_gui(
    system: FEMSystem,
    solver: BaseFEMSolver,
    cfg: FEMConfig,
    demo_name: str = "unknown",
    ui_cfg: GUIVisibilityConfig | None = None,
    mesh_presets: dict[str, dict] | None = None,
    rebuild_sim=None,
    logger: logging.Logger | None = None,
    debug: bool = False,
    defaults=None,
    scene_style: str = "softbody",
    cloth_density_range: tuple[int, int] | None = None,
    cloth_density_current: int | None = None,
    on_cloth_density_change=None,
    slider_ranges: dict[str, tuple[float, float]] | None = None,
    analysis_energy_fn: AnalysisEnergyFn = compute_model_total_energy,
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
    is_cloth = scene_style == "cloth"
    if is_cloth:
        cam_target = np.array([1.0, 1.2, 1.0], dtype=np.float32)
        cam_yaw = -1.25
        cam_pitch = 0.4
        cam_dist = 4.8
        cam_pos = np.array([3.5, 2.2, 4.8], dtype=np.float32)
    cloth_surface_mode = False

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
    softbody_scenario = "Side Stretch" if cfg.enable_side_stretch else "Default"
    current_mesh_name = next(iter(mesh_presets.keys())) if mesh_presets else ""
    if not current_mesh_name and cloth_density_current is not None:
        current_mesh_name = f"Density{int(cloth_density_current)}"
    target_fps = 120.0
    frame_dt = 1.0 / target_fps
    last_frame_t = time.perf_counter()
    frame_id = 0

    # Recording state
    _recording = False
    _record_dir = None
    _record_count = 0
    _record_frame = 0
    analysis_mode = False
    _log_sim_config(logger, "start", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)

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
            elif is_cloth and cloth_surface_mode:
                lmb_action = "camera"
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
                _log_sim_config(logger, "reset_sim", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
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
            new_analysis_mode = g.checkbox("Analysis Log", analysis_mode)
            if new_analysis_mode != analysis_mode:
                analysis_mode = new_analysis_mode
                _log_sim_config(
                    logger,
                    "analysis_on" if analysis_mode else "analysis_off",
                    demo_name,
                    scene_style,
                    current_mesh_name,
                    softbody_scenario,
                    system,
                    cfg,
                    solver,
                    analysis_mode,
                )
            if not is_cloth:
                g.text("Scenario:")
                if g.button("Default"):
                    softbody_scenario = "Default"
                    cfg.enable_side_stretch = False
                    cfg.enable_boundary_vibration = False
                    if cfg.constraint_mode != ConstraintMode.TOP:
                        cfg.constraint_mode = ConstraintMode.TOP
                        system.set_constraint_mode(ConstraintMode.TOP)
                        system.reset_state()
                    if logger is not None:
                        logger.info("Scenario switched: Default")
                    _log_sim_config(logger, "scenario_default", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
                if g.button("Side Stretch"):
                    softbody_scenario = "Side Stretch"
                    cfg.enable_side_stretch = True
                    cfg.enable_boundary_vibration = False
                    if cfg.constraint_mode != ConstraintMode.SIDE_X_BOTH:
                        cfg.constraint_mode = ConstraintMode.SIDE_X_BOTH
                        system.set_constraint_mode(ConstraintMode.SIDE_X_BOTH)
                        system.reset_state()
                    if logger is not None:
                        logger.info("Scenario switched: Side Stretch")
                    _log_sim_config(logger, "scenario_side_stretch", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
                g.text(f"Current: {softbody_scenario}")
                g.text("Constraint:")
                new_constraint_idx = constraint_idx
                for i, name in enumerate(_CONSTRAINT_NAMES):
                    if g.button(name):
                        new_constraint_idx = i
                if new_constraint_idx != constraint_idx:
                    constraint_idx = int(new_constraint_idx)
                    system.set_constraint_mode(ConstraintMode(constraint_idx))
                    system.reset_state()
                    _log_sim_config(logger, "constraint_change", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
                g.text(_CONSTRAINT_DESC.get(constraint_idx, ""))
            else:
                g.text("Constraint (Cloth):")
                if g.button("Single Corner"):
                    system.set_constraint_mode(ConstraintMode.SINGLE_CORNER)
                    system.reset_state()
                    _log_sim_config(logger, "constraint_single_corner", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
                if g.button("Two Inset Anchors"):
                    system.set_constraint_mode(ConstraintMode.TWO_CORNERS_INSET)
                    system.reset_state()
                    _log_sim_config(logger, "constraint_two_inset", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
                g.text(_CONSTRAINT_DESC.get(int(cfg.constraint_mode.value), ""))
                if cloth_density_range is not None and on_cloth_density_change is not None and cloth_density_current is not None:
                    dmin, dmax = cloth_density_range
                    new_density = g.slider_int("Density", int(cloth_density_current), dmin, dmax)
                    if int(new_density) != int(cloth_density_current):
                        system, solver = on_cloth_density_change(int(new_density))
                        cloth_density_current = int(new_density)
                        current_mesh_name = f"Density{cloth_density_current}"
                        constraint_idx = int(cfg.constraint_mode.value)
                        _log_sim_config(logger, "cloth_density_change", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)
            if mesh_presets and rebuild_sim is not None:
                g.text("Mesh:")
                g.text(f"Current: {current_mesh_name}")
                for name in mesh_presets.keys():
                    if g.button(name):
                        current_mesh_name = name
                        if logger is not None:
                            logger.info("Rebuild sim mesh=%s", name)
                        system, solver = rebuild_sim(name)
                        if is_cloth and cloth_density_current is not None and hasattr(system, "_nx"):
                            cloth_density_current = int(system._nx)
                        constraint_idx = int(cfg.constraint_mode.value)
                        _sync_collision_enabled(cfg, system)
                        _log_sim_config(logger, "mesh_change", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)

        # --- Controls panel ---
        p = ui_cfg.parameters
        if p.show_panel:
            with gui.sub_window("Controls", 0.72, 0.02, 0.26, 0.56) as panel:
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
                dt_min, dt_max = (1.0e-2, 5.0e-2)
                damp_min, damp_max = (0.90, 1.0)
                youngs_min, youngs_max = (1.0e3, 8.0e4)
                pois_min, pois_max = (0.05, 0.45)
                gy_min, gy_max = (-9.8, 0.0)
                if slider_ranges is not None:
                    dt_min, dt_max = slider_ranges.get("dt", (dt_min, dt_max))
                    damp_min, damp_max = slider_ranges.get("damping", (damp_min, damp_max))
                    youngs_min, youngs_max = slider_ranges.get("youngs", (youngs_min, youngs_max))
                    pois_min, pois_max = slider_ranges.get("poisson", (pois_min, pois_max))
                    gy_min, gy_max = slider_ranges.get("gravity_y", (gy_min, gy_max))
                if p.show_dt:
                    cfg.dt = panel.slider_float("dt", cfg.dt, dt_min, dt_max)
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
                    cfg.damping = panel.slider_float("damping", cfg.damping, damp_min, damp_max)
                if p.show_youngs:
                    cfg.youngs_modulus = panel.slider_float("Young's", cfg.youngs_modulus, youngs_min, youngs_max)
                if p.show_poisson:
                    cfg.poisson_ratio = panel.slider_float("Poisson", cfg.poisson_ratio, pois_min, pois_max)

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
                    gy = panel.slider_float("gravity_y", cfg.gravity[1], gy_min, gy_max)
                    cfg.gravity = (cfg.gravity[0], gy, cfg.gravity[2])

                _draw_collision_controls(panel, cfg, system, logger)

                # --- Boundary Vibration (B1 bonus) ---
                if p.show_boundary_vibration:
                    panel.text("--- Boundary Vibration ---")
                    cfg.enable_boundary_vibration = panel.checkbox("Vibrate fixed", cfg.enable_boundary_vibration)
                    if cfg.enable_boundary_vibration:
                        cfg.boundary_vibration_amplitude = panel.slider_float("Amplitude", cfg.boundary_vibration_amplitude, 0.0, 0.5)
                        cfg.boundary_vibration_frequency = panel.slider_float("Frequency", cfg.boundary_vibration_frequency, 0.5, 10.0)
                    cfg.enable_side_stretch = panel.checkbox("Side Stretch", cfg.enable_side_stretch)
                    if cfg.enable_side_stretch:
                        cfg.side_stretch_amplitude = panel.slider_float("Stretch Amp", cfg.side_stretch_amplitude, 0.0, 0.8)
                        cfg.side_stretch_frequency = panel.slider_float("Stretch Freq", cfg.side_stretch_frequency, 0.2, 10.0)

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
                        _log_sim_config(logger, "material_change", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)

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
                        _sync_collision_enabled(cfg, system)
                        if logger is not None:
                            logger.info("Reset to defaults")
                        _log_sim_config(logger, "reset_defaults", demo_name, scene_style, current_mesh_name, softbody_scenario, system, cfg, solver, analysis_mode)

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
                if is_cloth:
                    cloth_surface_mode = g.checkbox("Surface Cloth", cloth_surface_mode)

        solver_dt = 0.0
        if not paused:
            _sync_collision_enabled(cfg, system)
            step_t0 = time.perf_counter()
            solver.step()
            solver_dt = time.perf_counter() - step_t0
            if analysis_mode:
                _log_analysis_sample(
                    logger,
                    demo_name,
                    scene_style,
                    current_mesh_name,
                    softbody_scenario,
                    system,
                    cfg,
                    solver,
                    analysis_energy_fn,
                )
        scene.set_camera(camera)
        if show_lighting:
            scene.point_light(pos=(8, 12, 10), color=(1.0, 1.0, 1.0))
            scene.ambient_light((0.3, 0.3, 0.3))
        else:
            scene.ambient_light((1.0, 1.0, 1.0))
        if show_particles and not cloth_surface_mode:
            scene.particles(system.x, radius=0.05, color=(0.2, 0.7, 1.0))
        if is_cloth and cloth_surface_mode and hasattr(system, "tri_indices"):
            scene.mesh(vertices=system.x, indices=system.tri_indices, color=(0.45, 0.7, 0.9), two_sided=True)
        if show_wireframe:
            system.build_line_points()
            scene.lines(system.line_points, width=1.0, color=(0.85, 0.85, 0.9))

        # Render collision objects
        if hasattr(system, "collision_world") and system.collision_world is not None:
            cw = system.collision_world
            for plane in cw.planes:
                if not plane.enabled:
                    continue
                _draw_plane_grid(scene, plane.normal, plane.offset)
            for sphere in cw.spheres:
                if not sphere.enabled:
                    continue
                _draw_sphere_wire(scene, sphere.center, sphere.radius)
            for aabb in cw.aabbs:
                if not aabb.enabled:
                    continue
                _draw_aabb_wire(scene, aabb.bmin, aabb.bmax)

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

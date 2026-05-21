import taichi as ti

from lab3.constants import ConstraintMode, GUIVisibilityConfig, MaterialType, SoftBodyDefaults
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.logging_utils import create_lab3_logger
from lab3.solver import ImplicitNewtonCGSolver


def run(debug: bool = False, safe_boot: bool = False):
    logger = create_lab3_logger(debug)
    ti.init(arch=ti.cpu if safe_boot else ti.gpu)

    defaults = SoftBodyDefaults(
        gravity=(0.0, -0.05, 0.0),
        density=400.0,
        youngs_modulus=2.0e4,
        poisson_ratio=0.2,
        damping=0.995,
        dt=1.0e-2,
        substeps=5,
    )
    config = defaults.make_config(
        use_implicit=True,
        material_type=MaterialType.COROTATED,
        constraint_mode=ConstraintMode.TOP,
        newton_max_iters=8,
        cg_max_iters=50,
        newton_tol=1.0e-4,
        cg_tol=1.0e-6,
        hessian_fd_eps=1.0e-4,
    )
    mesh_presets = {
        # Ultra Low keeps the previous B4 default for quick implicit checks.
        "Ultra Low": {"wx": 4, "wy": 2, "wz": 2, "cell_size": 1.0},
        # Low/Med/High match the explicit softbody demos for comparison.
        "Low": {"wx": 8, "wy": 2, "wz": 2, "cell_size": 1.0},
        "Med": {"wx": 16, "wy": 4, "wz": 4, "cell_size": 0.5},
        "High": {"wx": 24, "wy": 6, "wz": 6, "cell_size": 1.0 / 3.0},
    }
    if safe_boot:
        mesh_presets = {"Ultra Low": mesh_presets["Ultra Low"]}
        config.newton_max_iters = min(config.newton_max_iters, 5)
        config.cg_max_iters = min(config.cg_max_iters, 30)
        logger.warning("Safe boot enabled: using CPU backend, Ultra Low mesh only, reduced Newton/CG iterations")

    config.wx, config.wy, config.wz = (
        mesh_presets["Ultra Low"]["wx"],
        mesh_presets["Ultra Low"]["wy"],
        mesh_presets["Ultra Low"]["wz"],
    )
    config.cell_size = mesh_presets["Ultra Low"]["cell_size"]

    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_material_text = False
    ui_cfg.parameters.show_boundary_vibration = False

    slider_ranges = {
        "dt": (1.0e-3, 2.0e-2),
        "damping": (0.95, 1.0),
        "youngs": (1.0e3, 5.0e4),
        "poisson": (0.05, 0.45),
        "gravity_y": (-2.0, 0.0),
    }

    def _rebuild(name: str):
        p = mesh_presets[name]
        config.wx, config.wy, config.wz, config.cell_size = p["wx"], p["wy"], p["wz"], p["cell_size"]
        logger.info("Rebuild b4 mesh=%s wx=%d wy=%d wz=%d", name, config.wx, config.wy, config.wz)
        s = FEMSystem(config)
        so = ImplicitNewtonCGSolver(s, config)
        return s, so

    system, solver = _rebuild("Ultra Low")
    run_gui(
        system,
        solver,
        config,
        demo_name="b4",
        ui_cfg=ui_cfg,
        mesh_presets=mesh_presets,
        rebuild_sim=_rebuild,
        logger=logger,
        debug=debug,
        defaults=defaults,
        slider_ranges=slider_ranges,
    )


if __name__ == "__main__":
    run()

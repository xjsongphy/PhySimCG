import taichi as ti

from lab3.constants import GUIVisibilityConfig, SoftBodyDefaults
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.logging_utils import create_lab3_logger
from lab3.solver import ExplicitFEMSolver, ImplicitNewtonCGSolver


def run(debug: bool = False, safe_boot: bool = False):
    logger = create_lab3_logger(debug)
    ti.init(arch=ti.cpu if safe_boot else ti.gpu)
    defaults = SoftBodyDefaults()
    config = defaults.make_config(
        use_implicit=False,
    )
    mesh_presets = {
        # Keep physical size fixed at 8x2x2 while increasing density.
        "Low": {"wx": 8, "wy": 2, "wz": 2, "cell_size": 1.0},
        "Med": {"wx": 16, "wy": 4, "wz": 4, "cell_size": 0.5},
        "High": {"wx": 24, "wy": 6, "wz": 6, "cell_size": 1.0 / 3.0},
    }
    if safe_boot:
        mesh_presets = {"Low": mesh_presets["Low"]}
        config.substeps = min(config.substeps, 3)
        logger.warning("Safe boot enabled: using CPU backend, Low mesh only, substeps<=3")
    config.wx, config.wy, config.wz = (
        mesh_presets["Low"]["wx"],
        mesh_presets["Low"]["wy"],
        mesh_presets["Low"]["wz"],
    )
    config.cell_size = mesh_presets["Low"]["cell_size"]
    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_material_dropdown = False
    ui_cfg.parameters.show_material_text = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False

    def _rebuild(name: str):
        p = mesh_presets[name]
        config.wx, config.wy, config.wz, config.cell_size = p["wx"], p["wy"], p["wz"], p["cell_size"]
        logger.info("Rebuild basic mesh=%s wx=%d wy=%d wz=%d", name, config.wx, config.wy, config.wz)
        s = FEMSystem(config)
        so = ImplicitNewtonCGSolver(s, config) if config.use_implicit else ExplicitFEMSolver(s, config)
        return s, so

    system, solver = _rebuild("Low")
    run_gui(
        system,
        solver,
        config,
        ui_cfg=ui_cfg,
        mesh_presets=mesh_presets,
        rebuild_sim=_rebuild,
        logger=logger,
        debug=debug,
        defaults=defaults,
    )


if __name__ == "__main__":
    run()

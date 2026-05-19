import taichi as ti

from lab3.cloth import ClothSystem
from lab3.constants import ClothDefaults, GUIVisibilityConfig, MaterialType
from lab3.gui import run_gui
from lab3.logging_utils import create_lab3_logger
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False, safe_boot: bool = False):
    logger = create_lab3_logger(debug)
    ti.init(arch=ti.cpu if safe_boot else ti.gpu)

    defaults = ClothDefaults()
    config = defaults.make_config(
        material_type=MaterialType.STVK,
        use_implicit=False,
    )
    mesh_presets = {
        "Low": {"nx": 16, "ny": 16},
        "Med": {"nx": 24, "ny": 24},
        "High": {"nx": 32, "ny": 32},
    }
    if safe_boot:
        mesh_presets = {"Low": mesh_presets["Low"]}
        config.substeps = min(config.substeps, 2)
        logger.warning("Safe boot enabled: using CPU backend, Low mesh only, substeps<=2")

    def _rebuild(name: str):
        p = mesh_presets[name]
        logger.info("Rebuild b2 mesh=%s nx=%d ny=%d", name, p["nx"], p["ny"])
        s = ClothSystem(config=config, nx=p["nx"], ny=p["ny"], sx=2.0, sy=2.0)
        so = ExplicitFEMSolver(s, config)
        return s, so

    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False
    # B2 baseline currently supports StVK cloth branch only.
    ui_cfg.parameters.show_material_dropdown = False
    ui_cfg.parameters.show_material_text = False

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
    )


if __name__ == "__main__":
    run()

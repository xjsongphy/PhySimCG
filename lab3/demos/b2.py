import taichi as ti

from lab3.cloth import ClothSystem
from lab3.constants import ClothDefaults, ConstraintMode, GUIVisibilityConfig, MaterialType
from lab3.gui import run_gui
from lab3.logging_utils import create_lab3_logger
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False, safe_boot: bool = False):
    logger = create_lab3_logger(debug)
    ti.init(arch=ti.cpu if safe_boot else ti.gpu)

    defaults = ClothDefaults()
    cloth_size = (2.8, 2.8)
    config = defaults.make_config(
        material_type=MaterialType.STVK,
        use_implicit=False,
        constraint_mode=ConstraintMode.SINGLE_CORNER,
    )
    mesh_presets = {
        "Low": {"nx": 10, "ny": 10},
        "Med": {"nx": 14, "ny": 14},
        "High": {"nx": 20, "ny": 20},
    }
    if safe_boot:
        mesh_presets = {"Low": mesh_presets["Low"]}
        config.substeps = min(config.substeps, 2)
        logger.warning("Safe boot enabled: using CPU backend, Low mesh only, substeps<=2")
    density_min, density_max = 8, 28
    cloth_density = mesh_presets["Low"]["nx"]

    def _rebuild(name: str):
        p = mesh_presets[name]
        logger.info("Rebuild b2 mesh=%s nx=%d ny=%d", name, p["nx"], p["ny"])
        s = ClothSystem(config=config, nx=p["nx"], ny=p["ny"], sx=cloth_size[0], sy=cloth_size[1])
        so = ExplicitFEMSolver(s, config)
        return s, so

    def _rebuild_density(n: int):
        nonlocal cloth_density
        cloth_density = max(density_min, min(density_max, int(n)))
        logger.info("Rebuild b2 density nx=ny=%d", cloth_density)
        s = ClothSystem(config=config, nx=cloth_density, ny=cloth_density, sx=cloth_size[0], sy=cloth_size[1])
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
    cloth_slider_ranges = {
        "dt": (5.0e-4, 5.0e-3),
        "damping": (0.95, 1.0),
        "youngs": (10.0, 1.0e3),
        "poisson": (0.05, 0.45),
        "gravity_y": (-15.0, 0.0),
    }
    run_gui(
        system,
        solver,
        config,
        demo_name="b2",
        ui_cfg=ui_cfg,
        mesh_presets=None,
        rebuild_sim=None,
        logger=logger,
        debug=debug,
        defaults=defaults,
        scene_style="cloth",
        cloth_density_range=(density_min, density_max),
        cloth_density_current=cloth_density,
        on_cloth_density_change=_rebuild_density,
        slider_ranges=cloth_slider_ranges,
    )


if __name__ == "__main__":
    run()

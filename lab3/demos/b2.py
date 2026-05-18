import taichi as ti

from lab3.cloth import ClothSystem
from lab3.constants import FEMConfig, GUIVisibilityConfig, MaterialType
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False):
    _ = debug
    ti.init(arch=ti.gpu)

    config = FEMConfig(
        gravity=(0.0, -9.8, 0.0),
        density=0.5,
        youngs_modulus=50.0,
        poisson_ratio=0.3,
        material_type=MaterialType.STVK,
        use_implicit=False,
        substeps=2,
        dt=1.0e-3,
        damping=0.995,
    )
    system = ClothSystem(config=config, nx=28, ny=28, sx=2.0, sy=2.0)
    solver = ExplicitFEMSolver(system, config)

    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False
    # B2 baseline currently supports StVK cloth branch only.
    ui_cfg.parameters.show_material_dropdown = False
    ui_cfg.parameters.show_material_text = False

    run_gui(system, solver, config, ui_cfg=ui_cfg)


if __name__ == "__main__":
    run()


import taichi as ti

from lab3.constants import FEMConfig, GUIVisibilityConfig
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver, ImplicitNewtonCGSolver


def run(debug: bool = False):
    _ = debug  # Reserved for future profiling overlay, keep interface aligned with lab2 demos.
    ti.init(arch=ti.gpu)
    config = FEMConfig(
        gravity=(0.0, -9.8, 0.0),
        use_implicit=False,
        substeps=2,
        dt=1.0e-3,
    )
    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_material_dropdown = False
    ui_cfg.parameters.show_material_text = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False

    system = FEMSystem(config)
    solver = ImplicitNewtonCGSolver(system, config) if config.use_implicit else ExplicitFEMSolver(system, config)
    run_gui(system, solver, config, ui_cfg=ui_cfg)


if __name__ == "__main__":
    run()

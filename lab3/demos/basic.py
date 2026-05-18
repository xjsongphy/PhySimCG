import taichi as ti

from lab3.constants import GUIVisibilityConfig, SoftBodyDefaults
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver, ImplicitNewtonCGSolver


def run(debug: bool = False):
    _ = debug  # Reserved for future profiling overlay, keep interface aligned with lab2 demos.
    ti.init(arch=ti.gpu)
    defaults = SoftBodyDefaults()
    config = defaults.make_config(
        use_implicit=False,
        substeps=2,  # keep basic faster while using soft-body defaults
    )
    mesh_presets = {
        "Low": {"wx": 6, "wy": 2, "wz": 2},
        "Med": {"wx": 8, "wy": 2, "wz": 2},
        "High": {"wx": 12, "wy": 3, "wz": 3},
    }
    config.wx, config.wy, config.wz = (
        mesh_presets["Med"]["wx"],
        mesh_presets["Med"]["wy"],
        mesh_presets["Med"]["wz"],
    )
    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_material_dropdown = False
    ui_cfg.parameters.show_material_text = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False

    def _rebuild(name: str):
        p = mesh_presets[name]
        config.wx, config.wy, config.wz = p["wx"], p["wy"], p["wz"]
        s = FEMSystem(config)
        so = ImplicitNewtonCGSolver(s, config) if config.use_implicit else ExplicitFEMSolver(s, config)
        return s, so

    system, solver = _rebuild("Med")
    run_gui(system, solver, config, ui_cfg=ui_cfg, mesh_presets=mesh_presets, rebuild_sim=_rebuild)


if __name__ == "__main__":
    run()

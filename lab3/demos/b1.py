import taichi as ti

from lab3.constants import GUIVisibilityConfig, SoftBodyDefaults
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False):
    _ = debug
    ti.init(arch=ti.gpu)

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
    config.wx, config.wy, config.wz = (
        mesh_presets["Low"]["wx"],
        mesh_presets["Low"]["wy"],
        mesh_presets["Low"]["wz"],
    )
    config.cell_size = mesh_presets["Med"]["cell_size"]

    ui_cfg = GUIVisibilityConfig()
    # B1 focuses on material comparison: keep material selector visible.
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False

    def _rebuild(name: str):
        p = mesh_presets[name]
        config.wx, config.wy, config.wz, config.cell_size = p["wx"], p["wy"], p["wz"], p["cell_size"]
        s = FEMSystem(config)
        so = ExplicitFEMSolver(s, config)
        return s, so

    system, solver = _rebuild("Low")
    run_gui(system, solver, config, ui_cfg=ui_cfg, mesh_presets=mesh_presets, rebuild_sim=_rebuild)


if __name__ == "__main__":
    run()

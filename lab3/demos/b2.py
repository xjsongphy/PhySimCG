import taichi as ti

from lab3.cloth import ClothSystem
from lab3.constants import ClothDefaults, GUIVisibilityConfig, MaterialType
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False):
    _ = debug
    ti.init(arch=ti.gpu)

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

    def _rebuild(name: str):
        p = mesh_presets[name]
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

    system, solver = _rebuild("Med")
    run_gui(system, solver, config, ui_cfg=ui_cfg, mesh_presets=mesh_presets, rebuild_sim=_rebuild)


if __name__ == "__main__":
    run()

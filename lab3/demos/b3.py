import taichi as ti

from lab3.collision import CollisionWorld
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
        enable_collision=True,
        collision_k=3.0e4,
        collision_c=120.0,
        collision_particle_radius=0.07,
        collision_iters=1,
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

    def _rebuild(name: str):
        p = mesh_presets[name]
        config.wx, config.wy, config.wz, config.cell_size = p["wx"], p["wy"], p["wz"], p["cell_size"]
        s = FEMSystem(config)
        w = CollisionWorld()
        # y=0 ground
        w.add_plane((0.0, 1.0, 0.0), 0.0, enabled=True)
        # x=0 wall
        w.add_plane((1.0, 0.0, 0.0), 0.0, enabled=True)
        # one analytic sphere
        w.add_sphere(center=(4.5, 0.8, 1.0), radius=0.65, enabled=True)
        # one AABB box
        w.add_aabb(bmin=(2.3, 0.0, 0.2), bmax=(3.0, 1.1, 1.3), enabled=True)
        s.set_collision_world(w)
        so = ExplicitFEMSolver(s, config)
        return s, so

    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False
    # Keep material selector visible for B3 experimentation.

    system, solver = _rebuild("Low")
    run_gui(system, solver, config, ui_cfg=ui_cfg, mesh_presets=mesh_presets, rebuild_sim=_rebuild)


if __name__ == "__main__":
    run()

import taichi as ti

from lab3.collision import CollisionWorld
from lab3.constants import FEMConfig, GUIVisibilityConfig
from lab3.core import FEMSystem
from lab3.gui import run_gui
from lab3.solver import ExplicitFEMSolver


def run(debug: bool = False):
    _ = debug
    ti.init(arch=ti.gpu)

    config = FEMConfig(
        gravity=(0.0, -0.05, 0.0),
        use_implicit=False,
        substeps=5,
        dt=1.0e-3,
        enable_collision=True,
        collision_k=3.0e4,
        collision_c=120.0,
        collision_particle_radius=0.07,
        collision_iters=1,
    )

    system = FEMSystem(config)
    world = CollisionWorld()
    # y=0 ground
    world.add_plane((0.0, 1.0, 0.0), 0.0, enabled=True)
    # x=0 wall
    world.add_plane((1.0, 0.0, 0.0), 0.0, enabled=True)
    # one analytic sphere
    world.add_sphere(center=(4.5, 0.8, 1.0), radius=0.65, enabled=True)
    # one AABB box
    world.add_aabb(bmin=(2.3, 0.0, 0.2), bmax=(3.0, 1.1, 1.3), enabled=True)
    system.set_collision_world(world)

    solver = ExplicitFEMSolver(system, config)

    ui_cfg = GUIVisibilityConfig()
    ui_cfg.parameters.show_implicit_toggle = False
    ui_cfg.parameters.show_newton_iters = False
    ui_cfg.parameters.show_cg_iters = False
    # Keep material selector visible for B3 experimentation.

    run_gui(system, solver, config, ui_cfg=ui_cfg)


if __name__ == "__main__":
    run()

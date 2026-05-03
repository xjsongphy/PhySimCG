import taichi as ti
from lab2.eulerian import EulerianSimulator
from lab2.gui import run_gui


def run(debug=False):
    ti.init(arch=ti.vulkan)

    nx, ny, nz = 32, 64, 32
    sim = EulerianSimulator(nx, ny, nz)
    sim.init_dam_break_density()
    sim.init_cell_types()

    def substep(dt, flip_ratio, gravity, **kw):
        sim.substep(dt=dt, gravity=gravity, **kw)
        sim.update_render_points()

    run_gui(
        sim,
        substep_fn=substep,
        dt=0.01,
        num_substeps=1,
        gravity=-9.8,
        window_title="Lab2 - B3 Eulerian Fluid",
        debug=debug,
    )


if __name__ == "__main__":
    run()

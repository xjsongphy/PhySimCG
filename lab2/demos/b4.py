import taichi as ti
from lab2.core import scene_particle_count
from lab2.apic import APICSimulator
from lab2.gui import run_gui


def run(debug=False):
    ti.init(arch=ti.gpu)

    nx, ny, nz = 24, 48, 24
    scene = "Dam Break"
    num_particles = scene_particle_count(scene, nx)

    sim = APICSimulator(nx, ny, nz, num_particles)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    run_gui(
        sim,
        substep_fn=sim.substep,
        dt=0.01,
        num_substeps=2,
        flip_ratio=0.95,
        gravity=-9.8,
        window_title="Lab2 - B4 APIC Fluid",
        debug=debug,
        show_obstacle=True,
        show_resolution=True,
        show_color=True,
        show_flip=True,
        show_solver=True,
        show_volume=True,
        show_debug_toggle=True,
    )


if __name__ == "__main__":
    run()

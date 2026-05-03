import taichi as ti
from lab2.core import scene_particle_count
from lab2.flip import FLIPSimulator
from lab2.gui import run_gui


def run():
    ti.init(arch=ti.vulkan)

    nx, ny, nz = 24, 48, 24
    scene = "Dam Break"
    num_particles = scene_particle_count(scene, nx)

    sim = FLIPSimulator(nx, ny, nz, num_particles)
    sim.init_dam_break()
    sim.init_cell_types()
    sim.init_colors()
    sim.update_particle_density()
    sim.store_initial_density()

    run_gui(
        sim,
        substep_fn=sim.substep,
        dt=0.02,
        num_substeps=2,
        flip_ratio=0.95,
        gravity=-9.8,
        window_title="Lab2 - FLIP Fluid",
    )


if __name__ == "__main__":
    run()

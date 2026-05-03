import taichi as ti
from lab2.flip import FLIPSimulator
from lab2.gui import run_gui


def run():
    ti.init(arch=ti.vulkan)

    # Grid resolution
    nx, ny, nz = 24, 48, 24

    # Estimate particle count for dam break
    dx = 1.0 / nx
    spacing = dx * 0.5
    lo_x, hi_x = 0.05, 0.45
    lo_y, hi_y = 0.05, 0.85
    lo_z, hi_z = 0.05, 0.45
    npx = int((hi_x - lo_x) / spacing)
    npy = int((hi_y - lo_y) / spacing)
    npz = int((hi_z - lo_z) / spacing)
    num_particles = npx * npy * npz

    sim = FLIPSimulator(nx, ny, nz, num_particles)

    # Initialize
    sim.init_dam_break()
    sim.init_cell_types()
    sim.update_particle_density()
    sim.store_initial_density()

    # Run with GUI
    run_gui(
        sim,
        substep_fn=sim.substep,
        dt=0.02,
        num_substeps=2,
        flip_ratio=0.95,
        gravity=-9.8,
        window_title="Lab2 - FLIP Fluid (Basic)",
    )


if __name__ == "__main__":
    run()

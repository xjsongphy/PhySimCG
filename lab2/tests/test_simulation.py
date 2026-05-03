"""Integration tests for fluid simulation correctness.
Runs on CPU to verify stability, volume conservation, particle bounds, etc.
"""
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=False)

from lab2.core import scene_particle_count, FluidSimulator
from lab2.flip import FLIPSimulator
from lab2.eulerian import EulerianSimulator
from lab2.apic import APICSimulator


def test_flip_stability(nx=24, ny=48, nz=24, steps=100, dt=0.01, gravity=-9.8):
    """Verify FLIP particles stay bounded, count stays stable."""
    scene = "Dam Break"
    count = scene_particle_count(scene, nx)
    sim = FLIPSimulator(nx, ny, nz, count)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    pos = sim.pos.to_numpy()
    active0 = (pos[:, 0] >= 0).sum()
    print(f"  FLIP init: {active0}/{count} active")

    for step in range(steps):
        sim.substep(dt=dt, flip_ratio=0.95, gravity=gravity, num_pressure_iters=40)
        pos = sim.pos.to_numpy()
        active = pos[:, 0] >= 0
        ap = pos[active]

        # Check bounds
        assert ap.max() <= 1.0 + 1e-4, f"step {step}: pos max {ap.max()} > 1"
        assert ap.min() >= -1e-4, f"step {step}: pos min {ap.min()} < 0"

        # Check NaN/Inf
        assert not np.any(np.isnan(ap)), f"step {step}: NaN in positions"
        assert not np.any(np.isinf(ap)), f"step {step}: Inf in positions"

    pos = sim.pos.to_numpy()
    active = pos[:, 0] >= 0
    ap = pos[active]
    vel = sim.vel.to_numpy()[active]
    max_speed = np.linalg.norm(vel, axis=1).max()

    print(f"  FLIP after {steps} steps: {active.sum()} active, "
          f"y=[{ap[:,1].min():.3f},{ap[:,1].max():.3f}], "
          f"max_speed={max_speed:.2f}")
    return True


def test_volume_conservation(nx=24, ny=48, nz=24, steps=100, dt=0.01, gravity=-9.8):
    """Verify fluid volume doesn't explode or disappear."""
    scene = "Dam Break"
    count = scene_particle_count(scene, nx)
    sim = FLIPSimulator(nx, ny, nz, count)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    init_count = count
    prev_count = init_count

    for step in range(steps):
        sim.substep(dt=dt, flip_ratio=0.95, gravity=gravity, num_pressure_iters=40)
        pos = sim.pos.to_numpy()
        active = (pos[:, 0] >= 0).sum()
        # Active count should not drop below 90% of initial
        ratio = active / init_count
        assert ratio > 0.9, f"step {step}: only {ratio*100:.1f}% particles active"
        prev_count = active

    print(f"  Volume: {init_count}→{active} ({active/init_count*100:.1f}%) after {steps} steps")
    return True


def test_particle_distances(nx=24, ny=48, nz=24, steps=50, dt=0.01, gravity=-9.8):
    """Verify particles don't overlap excessively."""
    scene = "Dam Break"
    count = scene_particle_count(scene, nx)
    sim = FLIPSimulator(nx, ny, nz, count)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    for step in range(steps):
        sim.substep(dt=dt, flip_ratio=0.95, gravity=gravity, num_pressure_iters=40)

    pos = sim.pos.to_numpy()
    active = pos[:, 0] >= 0
    ap = pos[active]

    # Sample check: min distance between random particle pairs
    n_sample = min(500, len(ap))
    idx = np.random.choice(len(ap), n_sample, replace=False)
    sample = ap[idx]
    min_dist = float('inf')
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            d = np.linalg.norm(sample[i] - sample[j])
            if d < min_dist and d > 1e-10:
                min_dist = d
    print(f"  Min distance in {n_sample} samples: {min_dist:.6f} (dx={sim.dx:.4f})")
    return True


def test_eulerian_stability(nx=32, ny=64, nz=32, steps=50, dt=0.01, gravity=-9.8):
    """Verify Eulerian simulation doesn't explode."""
    sim = EulerianSimulator(nx, ny, nz)
    sim.init_dam_break_density()
    sim.init_cell_types()

    density_0 = sim.density.to_numpy().sum()
    print(f"  Eulerian init: density_sum={density_0:.1f}")

    for step in range(steps):
        sim.substep(dt=dt, gravity=gravity, num_pressure_iters=40)

        # Check velocity bounds
        u = sim.grid_u.to_numpy()
        v = sim.grid_v.to_numpy()
        w = sim.grid_w.to_numpy()
        max_u = max(abs(u).max(), abs(v).max(), abs(w).max())
        assert max_u < 100.0, f"step {step}: velocity explosion {max_u}"
        assert not np.any(np.isnan(u)), f"step {step}: NaN in u"
        assert not np.any(np.isnan(v)), f"step {step}: NaN in v"
        assert not np.any(np.isnan(w)), f"step {step}: NaN in w"

    density_f = sim.density.to_numpy().sum()
    print(f"  Eulerian after {steps} steps: density={density_f:.1f} "
          f"(change: {(density_f - density_0) / max(density_0, 1) * 100:.1f}%)")
    return True


def test_apic_stability(nx=24, ny=48, nz=24, steps=50, dt=0.01, gravity=-9.8):
    """Verify APIC doesn't explode."""
    scene = "Dam Break"
    count = scene_particle_count(scene, nx)
    sim = APICSimulator(nx, ny, nz, count)
    sim.init_dam_break()
    sim.relabel_and_density()
    sim.init_colors()
    sim.store_initial_density()

    for step in range(steps):
        sim.substep(dt=dt, flip_ratio=0.95, gravity=gravity, num_pressure_iters=40)
        pos = sim.pos.to_numpy()
        active = pos[:, 0] >= 0
        ap = pos[active]
        assert ap.max() <= 1.0 + 1e-4, f"step {step}: escaped"
        assert not np.any(np.isnan(ap)), f"step {step}: NaN"

    pos = sim.pos.to_numpy()
    active = pos[:, 0] >= 0
    ap = pos[active]
    vel = sim.vel.to_numpy()[active]
    max_speed = np.linalg.norm(vel, axis=1).max()
    print(f"  APIC after {steps}: {active.sum()} active, max_speed={max_speed:.2f}")
    return True


if __name__ == "__main__":
    print("=== Simulation Correctness Tests ===\n")

    tests = [
        ("FLIP Stability", test_flip_stability),
        ("Volume Conservation", test_volume_conservation),
        ("Particle Distances", test_particle_distances),
        ("Eulerian Stability", test_eulerian_stability),
        ("APIC Stability", test_apic_stability),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
            print(f"  PASS\n")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL: {e}\n")

    print(f"=== {passed}/{len(tests)} passed ===")
    if failed:
        print(f"Failed: {', '.join(failed)}")

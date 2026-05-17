from dataclasses import dataclass
from enum import IntEnum


class MaterialType(IntEnum):
    STVK = 0
    NEO_HOOKEAN = 1
    COROTATED = 2


@dataclass(slots=True)
class FEMConfig:
    # Time integration
    dt: float = 1.0e-3
    substeps: int = 5
    damping: float = 0.995

    # Material / mass
    density: float = 400.0
    youngs_modulus: float = 2.0e4
    poisson_ratio: float = 0.2
    material_type: MaterialType = MaterialType.STVK

    # External acceleration
    gravity: tuple[float, float, float] = (0.0, -0.05, 0.0)

    # Mesh resolution for a box domain
    wx: int = 8
    wy: int = 2
    wz: int = 2
    cell_size: float = 1.0

    # Boundary condition: pin top layer by default
    pin_top_layer: bool = True

    # Solver control
    use_implicit: bool = False
    newton_max_iters: int = 12
    cg_max_iters: int = 80
    newton_tol: float = 1.0e-4
    cg_tol: float = 1.0e-6
    hessian_fd_eps: float = 1.0e-4

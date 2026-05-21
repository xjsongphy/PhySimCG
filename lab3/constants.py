from dataclasses import dataclass, field
from enum import IntEnum


class MaterialType(IntEnum):
    STVK = 0
    NEO_HOOKEAN = 1
    COROTATED = 2


class ConstraintMode(IntEnum):
    TOP = 0
    SIDE_X_MIN = 1
    SIDE_X_BOTH = 2
    TOP_BOTTOM = 3
    SINGLE_CORNER = 4
    TWO_CORNERS_INSET = 5


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
    constraint_mode: ConstraintMode = ConstraintMode.TOP

    # Solver control
    use_implicit: bool = False
    newton_max_iters: int = 12
    cg_max_iters: int = 80
    newton_tol: float = 1.0e-4
    cg_tol: float = 1.0e-6
    hessian_fd_eps: float = 1.0e-4

    # Collision (B3) - stage 1: config only, default disabled.
    enable_collision: bool = False
    collision_k: float = 2.0e4
    collision_c: float = 80.0
    collision_particle_radius: float = 0.06
    collision_iters: int = 1

    # Built-in ground plane (legacy/simple fallback). Disabled by default.
    enable_builtin_ground: bool = False
    builtin_ground_y: float = 0.0
    builtin_ground_restitution: float = 0.0

    # Cloth bending (B2/B3 cloth): simple edge-based bending regularization
    cloth_bend_k: float = 120.0
    cloth_bend_damping: float = 2.0
    # For TWO_CORNERS_INSET mode: inward offset ratio for each top corner.
    cloth_anchor_inset_ratio: float = 0.15
    # Initial sag amount for TWO_CORNERS_INSET mode as a fraction of cloth width.
    cloth_anchor_sag_ratio: float = 0.12

    # Boundary vibration (B1 bonus): fixed vertices sinusoidal vibration
    enable_boundary_vibration: bool = False
    boundary_vibration_amplitude: float = 0.1
    boundary_vibration_frequency: float = 2.0
    enable_side_stretch: bool = False
    side_stretch_amplitude: float = 0.2
    side_stretch_frequency: float = 2.0


@dataclass(slots=True)
class ParameterPanelVisibility:
    # Whole panel
    show_panel: bool = True
    # Controls in panel
    show_paused: bool = True
    show_implicit_toggle: bool = True
    show_dt: bool = True
    show_substeps: bool = True
    show_damping: bool = True
    show_youngs: bool = True
    show_poisson: bool = True
    show_gravity_y: bool = True
    show_newton_iters: bool = True
    show_cg_iters: bool = True
    show_material_dropdown: bool = True
    show_material_text: bool = True
    show_boundary_vibration: bool = False


@dataclass(slots=True)
class RenderPanelVisibility:
    # Render options grouped as another panel to avoid hard-coding in gui logic.
    show_panel: bool = True
    show_particles: bool = True
    show_wireframe: bool = True
    show_lighting: bool = True


@dataclass(slots=True)
class GUIVisibilityConfig:
    parameters: ParameterPanelVisibility = field(default_factory=ParameterPanelVisibility)
    render: RenderPanelVisibility = field(default_factory=RenderPanelVisibility)


@dataclass(slots=True)
class SoftBodyDefaults:
    gravity: tuple[float, float, float] = (0.0, -0.05, 0.0)
    density: float = 400.0
    youngs_modulus: float = 2.0e4
    poisson_ratio: float = 0.2
    damping: float = 0.995
    dt: float = 1.0e-3
    substeps: int = 5

    def make_config(self, **overrides) -> FEMConfig:
        cfg = FEMConfig(
            gravity=self.gravity,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            damping=self.damping,
            dt=self.dt,
            substeps=self.substeps,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass(slots=True)
class ClothDefaults:
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    density: float = 0.5
    youngs_modulus: float = 50.0
    poisson_ratio: float = 0.3
    damping: float = 0.995
    dt: float = 1.0e-3
    substeps: int = 2
    cloth_bend_k: float = 120.0
    cloth_bend_damping: float = 2.0

    def make_config(self, **overrides) -> FEMConfig:
        cfg = FEMConfig(
            gravity=self.gravity,
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            damping=self.damping,
            dt=self.dt,
            substeps=self.substeps,
            cloth_bend_k=self.cloth_bend_k,
            cloth_bend_damping=self.cloth_bend_damping,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

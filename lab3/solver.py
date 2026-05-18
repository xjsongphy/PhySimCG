from __future__ import annotations

import numpy as np

from lab3.constants import FEMConfig
from lab3.core import FEMSystem
from lab3.models import CorotatedModel, ElasticModel, NeoHookeanModel, StVKModel


def build_model(config: FEMConfig) -> ElasticModel:
    if config.material_type.value == 0:
        return StVKModel(config.youngs_modulus, config.poisson_ratio)
    if config.material_type.value == 1:
        return NeoHookeanModel(config.youngs_modulus, config.poisson_ratio)
    return CorotatedModel(config.youngs_modulus, config.poisson_ratio)


class BaseFEMSolver:
    def __init__(self, system: FEMSystem, config: FEMConfig):
        self.system = system
        self.config = config
        self.model = build_model(config)

    def set_material(self, model: ElasticModel) -> None:
        self.model = model

    def sync_material_from_config(self) -> None:
        self.model.youngs_modulus = self.config.youngs_modulus
        self.model.poisson_ratio = self.config.poisson_ratio
        self.model.mu, self.model.lmbda = self.model.to_lame(
            self.config.youngs_modulus, self.config.poisson_ratio
        )

    def step(self) -> None:
        raise NotImplementedError


class ExplicitFEMSolver(BaseFEMSolver):
    def substep(self) -> None:
        self.system.clear_forces()
        gx, gy, gz = self.config.gravity
        self.system.add_gravity(gx, gy, gz)
        self.system.accumulate_internal_forces(
            int(self.model.model_type.value), self.model.mu, self.model.lmbda
        )
        self.system.add_drag_force()
        if hasattr(self.system, "add_collision_forces"):
            self.system.add_collision_forces()
        self.system.integrate_explicit(self.config.dt, self.config.damping)
        if not self.config.enable_collision:
            self.system.apply_ground_plane(0.0, 0.0)

    def step(self) -> None:
        self.sync_material_from_config()
        for _ in range(self.config.substeps):
            self.substep()


class ImplicitNewtonCGSolver(BaseFEMSolver):
    def _evaluate_gradient(self, x: np.ndarray, y: np.ndarray, inv_dt2: float) -> np.ndarray:
        self.system.set_positions_numpy(x)
        f_int = self.system.evaluate_internal_force_numpy(
            int(self.model.model_type.value), self.model.mu, self.model.lmbda
        )
        m = self.system.get_masses_numpy()[:, None]
        grad = m * inv_dt2 * (x - y) - f_int
        fixed = self.system.get_fixed_numpy()
        grad[fixed] = 0.0
        return grad

    def _cg_solve(self, hvp, b: np.ndarray, x0: np.ndarray, tol: float, max_iters: int) -> np.ndarray:
        x = x0.copy()
        r = b - hvp(x)
        p = r.copy()
        rs_old = float(r @ r)
        if rs_old < tol * tol:
            return x

        for _ in range(max_iters):
            Ap = hvp(p)
            denom = float(p @ Ap) + 1.0e-12
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if rs_new < tol * tol:
                break
            beta = rs_new / (rs_old + 1.0e-12)
            p = r + beta * p
            rs_old = rs_new
        return x

    def step(self) -> None:
        self.sync_material_from_config()

        dt = self.config.dt
        inv_dt2 = 1.0 / (dt * dt)
        damping = self.config.damping
        g = np.array(self.config.gravity, dtype=np.float32)
        fixed = self.system.get_fixed_numpy()

        x_n = self.system.get_positions_numpy()
        v_n = self.system.get_velocities_numpy()
        y = x_n + dt * (v_n + dt * g[None, :])
        y[fixed] = x_n[fixed]

        x = x_n.copy()
        grad = self._evaluate_gradient(x, y, inv_dt2)

        for _ in range(self.config.newton_max_iters):
            free_mask = ~fixed
            grad_norm = float(np.linalg.norm(grad[free_mask])) if free_mask.any() else 0.0
            if grad_norm < self.config.newton_tol:
                break

            grad_flat = grad.reshape(-1)
            eps = self.config.hessian_fd_eps

            def hvp(p_flat: np.ndarray) -> np.ndarray:
                p = p_flat.reshape((-1, 3)).copy()
                p[fixed] = 0.0
                x_perturb = x + eps * p
                grad_perturb = self._evaluate_gradient(x_perturb, y, inv_dt2)
                hv = (grad_perturb - grad) / eps
                hv[fixed] = 0.0
                return hv.reshape(-1)

            delta_flat = self._cg_solve(
                hvp=hvp,
                b=-grad_flat,
                x0=np.zeros_like(grad_flat),
                tol=self.config.cg_tol,
                max_iters=self.config.cg_max_iters,
            )
            delta = delta_flat.reshape((-1, 3))
            delta[fixed] = 0.0

            alpha = 1.0
            accepted = False
            for _ in range(8):
                x_trial = x + alpha * delta
                x_trial[fixed] = x_n[fixed]
                grad_trial = self._evaluate_gradient(x_trial, y, inv_dt2)
                if np.linalg.norm(grad_trial[free_mask]) <= np.linalg.norm(grad[free_mask]):
                    x = x_trial
                    grad = grad_trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break

        v_new = damping * (x - x_n) / dt
        v_new[fixed] = 0.0

        self.system.set_positions_numpy(x)
        self.system.set_velocities_numpy(v_new)
        self.system.apply_ground_plane(0.0, 0.0)

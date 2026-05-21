from abc import ABC
import taichi as ti

from lab3.constants import MaterialType


class ElasticModel(ABC):
    model_type: MaterialType

    def __init__(self, youngs_modulus: float, poisson_ratio: float):
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.mu, self.lmbda = self.to_lame(youngs_modulus, poisson_ratio)

    @staticmethod
    def to_lame(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
        mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        lmbda = (youngs_modulus * poisson_ratio) / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )
        return mu, lmbda


class StVKModel(ElasticModel):
    model_type = MaterialType.STVK


class NeoHookeanModel(ElasticModel):
    model_type = MaterialType.NEO_HOOKEAN


class CorotatedModel(ElasticModel):
    model_type = MaterialType.COROTATED


@ti.func
def stvk_first_piola(F, mu, lmbda):
    I = ti.Matrix.identity(ti.f32, 3)
    G = 0.5 * (F.transpose() @ F - I)
    trG = G.trace()
    S = 2.0 * mu * G + lmbda * trG * I
    return F @ S


@ti.func
def neo_hookean_first_piola(F, mu, lmbda):
    J = F.determinant()
    eps = 1.0e-8
    J_safe = ti.max(J, eps)
    FinvT = F.inverse().transpose()
    return mu * (F - FinvT) + lmbda * ti.log(J_safe) * FinvT


@ti.func
def corotated_first_piola(F, mu, lmbda):
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    J = F.determinant()
    FinvT = F.inverse().transpose()
    # Common robust corotated force form
    return 2.0 * mu * (F - R) + lmbda * (J - 1.0) * J * FinvT


@ti.func
def stvk_first_piola_tri(F, mu, lmbda):
    I2 = ti.Matrix.identity(ti.f32, 2)
    G = 0.5 * (F.transpose() @ F - I2)
    trG = G.trace()
    S = 2.0 * mu * G + lmbda * trG * I2
    return F @ S


@ti.func
def stvk_energy_density(F, mu, lmbda):
    I = ti.Matrix.identity(ti.f32, 3)
    E = 0.5 * (F.transpose() @ F - I)
    tr_e = E.trace()
    return mu * (E * E).sum() + 0.5 * lmbda * tr_e * tr_e


@ti.func
def neo_hookean_energy_density(F, mu, lmbda):
    J = F.determinant()
    J_safe = ti.max(J, 1.0e-8)
    log_j = ti.log(J_safe)
    return 0.5 * mu * ((F * F).sum() - 3.0) - mu * log_j + 0.5 * lmbda * log_j * log_j


@ti.func
def corotated_energy_density(F, mu, lmbda):
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    J = F.determinant()
    return mu * ((F - R) * (F - R)).sum() + 0.5 * lmbda * (J - 1.0) * (J - 1.0)


@ti.func
def stvk_energy_density_tri(F, mu, lmbda):
    I2 = ti.Matrix.identity(ti.f32, 2)
    E = 0.5 * (F.transpose() @ F - I2)
    tr_e = E.trace()
    return mu * (E * E).sum() + 0.5 * lmbda * tr_e * tr_e

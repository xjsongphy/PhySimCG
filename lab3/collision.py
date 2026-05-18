from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(slots=True)
class PlaneCollider:
    # Plane equation: dot(n, x) - offset = 0
    normal: np.ndarray
    offset: float
    enabled: bool = True

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float32)
        nrm = np.linalg.norm(n)
        if nrm < 1.0e-8:
            raise ValueError("Plane normal must be non-zero.")
        self.normal = n / nrm
        self.offset = float(self.offset)

    def query(self, x: np.ndarray, radius: float) -> tuple[bool, np.ndarray, float]:
        # Signed distance to plane (positive above plane in normal direction).
        signed = float(np.dot(self.normal, x) - self.offset)
        penetration = radius - signed
        if penetration > 0.0:
            return True, self.normal, penetration
        return False, self.normal, 0.0


@dataclass(slots=True)
class SphereCollider:
    center: np.ndarray
    radius: float
    enabled: bool = True

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError("Sphere radius must be > 0.")

    def query(self, x: np.ndarray, particle_radius: float) -> tuple[bool, np.ndarray, float]:
        d = x - self.center
        dist = float(np.linalg.norm(d))
        total = self.radius + particle_radius
        if dist < total:
            if dist > 1.0e-8:
                n = d / dist
            else:
                n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            return True, n.astype(np.float32), total - dist
        return False, np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0


@dataclass(slots=True)
class AABBCollider:
    bmin: np.ndarray
    bmax: np.ndarray
    enabled: bool = True

    def __post_init__(self):
        self.bmin = np.asarray(self.bmin, dtype=np.float32)
        self.bmax = np.asarray(self.bmax, dtype=np.float32)
        if np.any(self.bmax <= self.bmin):
            raise ValueError("AABB must satisfy bmax > bmin on every axis.")

    def query(self, x: np.ndarray, particle_radius: float) -> tuple[bool, np.ndarray, float]:
        # Treat point as a sphere; check against expanded AABB.
        emn = self.bmin - particle_radius
        emx = self.bmax + particle_radius
        inside = np.all(x >= emn) and np.all(x <= emx)
        if not inside:
            return False, np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.0

        # Push to nearest face of expanded box.
        dists = np.array(
            [
                x[0] - emn[0],  # -x face
                emx[0] - x[0],  # +x face
                x[1] - emn[1],  # -y face
                emx[1] - x[1],  # +y face
                x[2] - emn[2],  # -z face
                emx[2] - x[2],  # +z face
            ],
            dtype=np.float32,
        )
        face = int(np.argmin(dists))
        penetration = float(dists[face])
        if face == 0:
            n = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif face == 1:
            n = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif face == 2:
            n = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        elif face == 3:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif face == 4:
            n = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return True, n, penetration


class CollisionWorld:
    def __init__(self):
        self.planes: List[PlaneCollider] = []
        self.spheres: List[SphereCollider] = []
        self.aabbs: List[AABBCollider] = []

    def clear(self) -> None:
        self.planes.clear()
        self.spheres.clear()
        self.aabbs.clear()

    def add_plane(self, normal, offset: float, enabled: bool = True) -> None:
        self.planes.append(PlaneCollider(normal=np.asarray(normal, dtype=np.float32), offset=offset, enabled=enabled))

    def add_sphere(self, center, radius: float, enabled: bool = True) -> None:
        self.spheres.append(SphereCollider(center=np.asarray(center, dtype=np.float32), radius=radius, enabled=enabled))

    def add_aabb(self, bmin, bmax, enabled: bool = True) -> None:
        self.aabbs.append(AABBCollider(bmin=np.asarray(bmin, dtype=np.float32), bmax=np.asarray(bmax, dtype=np.float32), enabled=enabled))

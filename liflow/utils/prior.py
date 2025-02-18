import abc

import numpy as np
from ase import units


class Prior(abc.ABC):
    @abc.abstractmethod
    def sample(self, shape: tuple) -> np.ndarray:
        pass


class NormalPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple) -> np.ndarray:
        return self.rng.normal(scale=self.scale, size=shape)


class UniformScaleNormalPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple) -> np.ndarray:
        scale = self.rng.uniform(0, self.scale, size=shape[0])
        return self.rng.normal(scale=scale[:, None], size=shape)


class MaxwellBoltzmannPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(
        self, temperature: float, masses: np.ndarray, shape: tuple
    ) -> np.ndarray:
        scale = self.scale * np.sqrt(units.kB * temperature / masses)
        return self.rng.normal(scale=scale[:, None], size=shape)


class AdaptiveMaxwellBoltzmannPrior:
    def __init__(
        self,
        scale: list[list[float]] = [[1.0, 10.0], [1.0, 5.0]],  # [Li_scale, prior_scale]
        seed: int = 42,
    ):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        temperature: float,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        scale_Li_index: int,
        scale_frame_index: int,
        shape: tuple,
    ) -> np.ndarray:
        scale_Li = self.scale[0][scale_Li_index]
        scale_frame = self.scale[1][scale_frame_index]
        prefactor = np.where(atomic_numbers == 3, scale_Li, scale_frame)
        scale = prefactor * np.sqrt(units.kB * temperature / masses)
        return self.rng.normal(scale=scale[:, None], size=shape)


def get_prior(class_name: str, **kwargs):
    return {
        "NormalPrior": NormalPrior,
        "UniformScaleNormalPrior": UniformScaleNormalPrior,
        "MaxwellBoltzmannPrior": MaxwellBoltzmannPrior,
        "AdaptiveMaxwellBoltzmannPrior": AdaptiveMaxwellBoltzmannPrior,
    }[class_name](**kwargs)

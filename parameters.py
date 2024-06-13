from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    S0: float = 100
    K: float = 110
    mu: float = 0.07
    r: float = 0.02
    sigma: float = 0.3
    T: float = 1
    n: int = 100
    num_paths: int = 50000
    dt: float = None
    u: float = None
    d: float = None
    p: float = None

    def __post_init__(self):
        """Calculate u, d, p based on T, r, sigma, and num_paths"""
        self.dt = self.T / self.n
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(self.mu * self.dt) - self.d) / (self.u - self.d)


def parameters_default():
    return Parameters(S0=100, K=110, mu=0.07, r=0.02, sigma=0.3, T=1, n=100, num_paths=50000)

# longstaff schwartz paper:


def parameters_longstaff():
    return Parameters(S0=36, K=40, mu=0.06, r=0.06, sigma=0.2, T=1, n=50, num_paths=5000)


def parameters_stylized_longstaff():
    return Parameters(S0=1, K=1.1, mu=0.06, r=0.06, sigma=0.2, T=3, n=3, num_paths=5000)


def paths_longstaff():
    return np.array([[1.00, 1.09, 1.08, 1.34],
                     [1.00, 1.16, 1.26, 1.54],
                     [1.00, 1.22, 1.07, 1.03],
                     [1.00, 0.93, 0.97, 0.92],
                     [1.00, 1.11, 1.56, 1.52],
                     [1.00, 0.76, 0.77, 0.90],
                     [1.00, 0.92, 0.84, 1.01],
                     [1.00, 0.88, 1.22, 1.34]])

"""Cruce con una llegada Bernoulli independiente por vía y paso."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Cruce(gym.Env):
    def __init__(self, horizonte=160):
        super().__init__()
        self.horizonte = horizonte
        self.observation_space = spaces.MultiDiscrete([5, 5, 2])
        self.action_space = spaces.Discrete(2)

    def estado(self):
        return np.array([min(4, self.colas[0]), min(4, self.colas[1]), self.fase])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.paso = self.fase = 0
        self.colas = [0, 0]
        return self.estado(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Acción esperada: 0 (mantener) o 1 (cambiar).")
        self.fase ^= int(action)
        llegadas = self.np_random.random(2) < (0.45, 0.30)
        for via in range(2):
            self.colas[via] += int(llegadas[via])
        self.colas[self.fase] = max(0, self.colas[self.fase] - 1)
        espera = sum(self.colas)
        self.paso += 1
        info = {"auto_pasos": espera, "llegadas": int(llegadas.sum())}
        return self.estado(), -espera - int(action), False, self.paso >= self.horizonte, info

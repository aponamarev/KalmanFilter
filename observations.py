import numpy as np


class Observations(object):

    def __init__(self, 
        start: float=2.0, 
        decay: float=0.975, 
        cosine_step: float = 0.05, 
        noise_x: float = 0.5,
        noise_v: float = 0.5):

        # ensure consistency of experimentation
        np.random.seed(1)
        
        self.s: float = start
        self.decay: float = decay
        self.step: float = cosine_step
        self.n: int = 0
        self.noise_x: float = noise_x
        self.noise_v: float = noise_v
        self.x0: float = 0.0

    def get(self) -> tuple:
        """Returns an observations

        Returns:
            tuple: (x, v)
        """
        self.n += 1
        step = np.pi * self.n * self.step
        cos = 1 + np.cos(step)
        x = self.s * self.decay**self.n + cos
        x_t0 = self.x0
        v = x-x_t0

        self.x0 = x

        x += self.noise_x * (np.random.random() - 0.5)
        v += self.noise_v * (np.random.random() - 0.5)

        return x, v
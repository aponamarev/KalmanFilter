import numpy as np


class Observations(object):

    def __init__(self, 
        start: float=2.0, 
        decay: float=0.9, 
        cosine_step: float = 0.1, 
        noise: float = 0.5):
        
        self.s: float = start
        self.decay: float = decay
        self.step: float = cosine_step
        self.n: int = 0
        self.noise: float = noise
        self.x0: float = 0.0

    def get(self) -> tuple:
        """Returns an observations

        Returns:
            tuple: (x, v)
        """
        self.n += 1
        step = self.n * self.step
        step = np.pi/2 * (step-int(step))
        noise = np.random.random() * self.noise
        x = self.s * (0.5+np.cos(step)) * self.decay**self.n
        x_t0 = self.x0
        v = x-x_t0
        v_noise = 1-0.1*(np.random.random()-0.5)

        self.x0 = x

        return x + noise, v * v_noise
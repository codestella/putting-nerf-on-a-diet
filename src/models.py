import jax
import haiku as hk
import jax.numpy as np


class Model(hk.Module):
    def __init__(self):
        super().__init__()
        self.width = 256
        self.depth = 6
        self.use_viewdirs = False

    def __call__(self, coords):
        sh = coords.shape
        coords = np.reshape(coords, [-1 ,3])

        # positional encoding
        x = np.concatenate([np.concatenate([
            np.sin(coords*(2**i)), np.cos(coords*(2**i))
        ], axis=-1) for i in np.linspace(0, 8, 20)], axis=-1)

        for _ in range(self.depth-1):
            x = hk.Linear(output_size=self.width)(x)
            x = jax.nn.relu(x)

        out = hk.Linear(output_size=4)(x)
        out = np.reshape(out, list(sh[:-1]) + [4])
        return out

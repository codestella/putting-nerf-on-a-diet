import haiku as hk
from haiku._src import utils
import jax.numpy as np
import jax

class Model(hk.Module):
    def __init__(self):
        super().__init__()
        self.width = 256
        self.depth = 6
        self.use_viewdirs = False

    def __call__(self, coords, view_dirs=None):
        sh = coords.shape
        if self.use_viewdirs:
            viewdirs = None
            viewdirs = np.repeat(viewdirs[..., None, :], coords.shape[-2], axis=-2)
            viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
            viewdirs = np.reshape(viewdirs, (-1, 3))
            viewdirs = hk.Linear(output_size=self.width // 2)(viewdirs)
            viewdirs = jax.nn.relu(viewdirs)
        coords = np.reshape(coords, [-1, 3])

        x = np.concatenate([np.concatenate([np.sin(coords * (2 ** i)), np.cos(coords * (2 ** i))], axis=-1) for i in
                            np.linspace(0, 8, 20)], axis=-1)

        for _ in range(self.depth - 1):
            x = hk.Linear(output_size=self.width)(x)
            x = jax.nn.relu(x)

        if self.use_viewdirs:
            density = hk.Linear(output_size=1)(x)
            x = np.concatenate([x, viewdirs], axis=-1)
            x = hk.Linear(output_size=self.width)(x)
            x = jax.nn.relu(x)
            rgb = hk.Linear(output_size=3)(x)
            out = np.concatenate([density, rgb], axis=-1)
        else:
            out = hk.Linear(output_size=4)(x)
        out = np.reshape(out, list(sh[:-1]) + [4])
        return out
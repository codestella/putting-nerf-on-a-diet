import jax
import jax.numpy as np
import flax.linen as nn


class Model(nn.Module):   
    width = 256
    depth = 6

    @nn.compact
    def __call__(self, coords):
        sh = coords.shape
        coords = np.reshape(coords, [-1 ,3])

        # positional encoding
        x = np.concatenate([np.concatenate([
            np.sin(coords*(2**i)), np.cos(coords*(2**i))
        ], axis=-1) for i in np.linspace(0, 8, 20)], axis=-1)

        for idx in range(self.depth - 1):
            x = nn.Dense(self.width, name=f"fc{idx}")(x)
            x = nn.relu(x)

        out = nn.Dense(4, name="fc_last")(x)
        out = np.reshape(out, list(sh[:-1]) + [4])
        return out

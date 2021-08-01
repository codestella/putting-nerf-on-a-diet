import os
import flax
from jax import random
from flax.training import checkpoints

from nerf import models
from nerf import utils
from demo.src.config import NerfConfig

rng = random.PRNGKey(0)
# TODO @Alex: make image size flexible if needed
dummy_rays = random.uniform(rng, shape=NerfConfig.IMAGE_SHAPE)
dummy_batch = {"rays": utils.Rays(dummy_rays, dummy_rays, dummy_rays)}
dummy_lr = 1e-2


def load_trained_model(model_dir, model_fn):
    model, init_variables = init_model()
    optimizer = flax.optim.Adam(dummy_lr).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables
    assert os.path.isfile(os.path.join(model_dir, model_fn))
    state = checkpoints.restore_checkpoint(model_dir, state,
                                           prefix=model_fn)
    return model, state


def init_model():
    _, key = random.split(rng)
    model, init_variables = models.get_model(key, dummy_batch,
                                             NerfConfig)
    return model, init_variables


if __name__ == '__main__':
    _model_dir = '../ship_fewshot_wsc'
    _model_fn = 'checkpoint_345000'
    _model, _state = load_trained_model(_model_dir, _model_fn)

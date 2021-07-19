import math
from typing import Optional
from absl import flags
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from transformers import FlaxCLIPModel

FLAGS = flags.FLAGS

@partial(jax.jit, static_argnums=[0, 1])
def update_semantic_loss(model, clip_model, rng, state, batch, lr, gamma):
    # the batch is without shard
    random_rays = batch["random_rays"]
    #rng, key_0, key_1 = rng
    rng, key_0, key_1 = random.split(rng,3)

    def semantic_loss(variables):
        src_ret = model.apply(variables, key_0, key_1, random_rays, False)
        src_image = src_ret[-1][0]
        # reshape flat pixel to an image (assume 3 channels & square shape)
        w = int(math.sqrt(src_image.shape[0]))
        src_image = src_image.reshape([w, w, 3])
        src_embedding = clip_model.get_image_features(pixel_values=preprocess_for_CLIP(jnp.expand_dims(src_image,0).transpose(0, 3, 1, 2)))
        src_embedding /= jnp.linalg.norm(src_embedding, axis=-1, keepdims=True)
        src_embedding = jnp.array(src_embedding)
        target_embedding = batch["embedding"]
        sc_loss = 0.5 * jnp.sum((src_embedding - target_embedding) ** 2) / src_embedding.shape[0]
        return sc_loss * gamma, src_image

    (sc_loss, src_image), grad = jax.value_and_grad(semantic_loss, has_aux = True)(jax.device_get(jax.tree_map(lambda x:x[0], state)).optimizer.target)
    return sc_loss, grad, src_image

def trans_t(t):
    return jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=jnp.float32)

def rot_phi(phi):
    return jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(phi), -np.sin(phi), 0],
        [0, jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)

def rot_theta(th):
    return jnp.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, jnp.cos(th), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)

def pose_spherical(radius, theta, phi):
    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

def random_pose(rng, bds):
    rng, *rng_inputs = jax.random.split(rng, 3)
    radius = random.uniform(rng_inputs[1], minval=bds[0], maxval=bds[1])
    theta = random.uniform(rng_inputs[1], minval=0, maxval=jnp.pi/2)
    phi = random.uniform(rng_inputs[1], minval=0, maxval=jnp.pi*2)
    return pose_spherical(radius, theta, phi)

def preprocess_for_CLIP(image):
    """
    jax-based preprocessing for CLIP
    image  [B, 3, H, W]: batch image
    return [B, 3, 224, 224]: pre-processed image for CLIP
    """
    B, D, H, W = image.shape
    image = jax.image.resize(image, (B, D, 224, 224), 'bicubic')  # assume that images have rectangle shape.
    mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
    std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
    image = (image - mean.astype(image.dtype)) / std.astype(image.dtype)
    return image

def init_CLIP(dtype: str, model_name: Optional[str]) -> FlaxCLIPModel:
    if dtype == 'float16':
        dtype = jnp.float16
    elif dtype == 'float32':
        dtype = jnp.float32
    else:
        raise ValueError

    if model_name is None:
        model_name = 'openai/clip-vit-base-patch32'
    return FlaxCLIPModel.from_pretrained(model_name, dtype=dtype)

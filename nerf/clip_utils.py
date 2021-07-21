import math
from typing import Optional
from absl import flags
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from transformers import FlaxCLIPModel

from nerf import utils

FLAGS = flags.FLAGS

@partial(jax.jit, static_argnums=[0])
def semantic_loss(clip_model, src_image, target_embedding): 
    # c_image = utils.unshard(src_image[0])
    f_image = utils.unshard(src_image[1])

    w = int(math.sqrt(src_image[1].size//3))
    # c_image = c_image.reshape([w, w, 3])
    f_image = f_image.reshape([w, w, 3])
 
    src_embedding = clip_model.get_image_features(pixel_values=preprocess_for_CLIP(jnp.expand_dims(f_image,0).transpose(0, 3, 1, 2)))
    # src_embedding = clip_model.get_image_features(pixel_values=preprocess_for_CLIP(jnp.stack([c_image, f_image]).transpose(0, 3, 1, 2)))
    src_embedding /= jnp.linalg.norm(src_embedding, axis=-1, keepdims=True)
    sc_loss = 0.5 * jnp.sum((src_embedding - target_embedding) ** 2)
    return sc_loss, f_image

def semantic_step_multi(render_pfn, clip_model, rng, state, batch, lr):
    random_rays = jax.tree_map(lambda x: utils.shard(x).astype(jnp.float16), batch["random_rays"])
    target_embedding = batch["embedding"].astype(jnp.float16)
    rng, key_0, key_1 = random.split(rng,3)

    def loss_fn(variables):
        src_image = render_pfn(variables, key_0, key_1, random_rays)
        sc_loss, f_image = semantic_loss(clip_model, src_image, target_embedding)
        return sc_loss * FLAGS.sc_loss_mult, f_image
    (sc_loss, src_image), grad = jax.value_and_grad(loss_fn, has_aux = True)(jax.device_get(jax.tree_map(lambda x:x[0], state)).optimizer.target)
    return sc_loss, grad, src_image

@partial(jax.jit, static_argnums=[0, 1])
def semantic_step_single(model, clip_model, rng, state, batch, lr):
    batch = jax.tree_map(lambda x: x.astype(jnp.float16), batch)
    # the batch is without shard
    random_rays = batch["random_rays"]
    rng, key_0, key_1 = random.split(rng,3)

    def semantic_loss(variables):
        c_image, f_image = model.apply(variables, key_0, key_1, random_rays, False, rgb_only = True)
        # reshape flat pixel to an image (assume 3 channels & square shape)
        w = int(math.sqrt(f_image.shape[0]))
        # c_image = c_image.reshape([w, w, 3])
        f_image = f_image.reshape([w, w, 3])

        src_embedding = clip_model.get_image_features(pixel_values=preprocess_for_CLIP(jnp.expand_dims(f_image,0).transpose(0, 3, 1, 2)))
        # src_embedding = clip_model.get_image_features(pixel_values=preprocess_for_CLIP(jnp.stack([c_image, f_image]).transpose(0, 3, 1, 2)))
        src_embedding /= jnp.linalg.norm(src_embedding, axis=-1, keepdims=True)
        src_embedding = jnp.array(src_embedding)
        target_embedding = batch["embedding"]
        sc_loss = 0.5 * jnp.sum((src_embedding - target_embedding)**2)
        return sc_loss * FLAGS.sc_loss_mult, f_image
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
        [0, jnp.cos(phi), jnp.sin(phi), 0],
        [0,-jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)

def rot_theta(th):
    return jnp.array([
        [jnp.cos(th), 0,-jnp.sin(th), 0],
        [0, 1, 0, 0],
        [jnp.sin(th), 0, jnp.cos(th), 0],
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
    theta = random.uniform(rng_inputs[1], minval=-jnp.pi, maxval=jnp.pi)
    phi = random.uniform(rng_inputs[1], minval=0, maxval=jnp.pi/2)
    return pose_spherical(radius, theta, phi)

def preprocess_for_CLIP(image):
    """
    jax-based preprocessing for CLIP
    image  [B, 3, H, W]: batch image
    return [B, 3, 224, 224]: pre-processed image for CLIP
    """
    B, D, H, W = image.shape
    mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1)
    std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1)
    image = jax.image.resize(image, (B, D, 224, 224), 'bicubic')  # assume that images have rectangle shape.
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

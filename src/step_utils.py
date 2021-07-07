import jax
import jax.numpy as np
from jax import jit, random

import jmp
my_policy = jmp.Policy(compute_dtype=np.float16,
                       param_dtype=np.float16,
                       output_dtype=np.float16)

def render_fn(rnd_input, model, params, bvals, rays, near, far, N_samples, rand):
    chunk = 5
    for i in range(0, rays.shape[1], chunk):
        out = render_fn_inner(rnd_input, model, params, bvals, rays[:, i:i + chunk], near, far, rand, True, N_samples)
        if i == 0:
            rets = out
        else:
            rets = [np.concatenate([a, b], 0) for a, b in zip(rets, out)]
    return rets


def render_fn_inner(rnd_input, model, params, bvals, rays, near, far, rand, allret, N_samples):
    return render_rays(rnd_input, model, params, bvals, rays, near, far,
                       N_samples=N_samples, rand=rand, allret=allret)


def render_rays(rnd_input, model, params,
                bvals, rays, near, far,
                N_samples, rand=False, allret=False):
    rays_o, rays_d = rays

    # Compute 3D query points
    z_vals = np.linspace(near, far, N_samples, dtype = rays.dtype)
    if rand:
        z_vals += random.uniform(rnd_input, shape=list(rays_o.shape[:-1]) + [N_samples], dtype = rays.dtype) * (far - near) / N_samples
    # r(t) = o + t*d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = np.reshape(pts, [-1, 3])
    if bvals is not None:
        pts_flat = np.concatenate([np.sin(pts_flat @ bvals.T),
                                   np.cos(pts_flat @ bvals.T)], axis=-1)

    raw = model.apply(params, pts_flat)
    raw = my_policy.cast_to_compute((raw))
    raw = np.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    rgb, sigma_a = raw[..., :3], raw[..., 3]
    sigma_a = jax.nn.relu(sigma_a)
    rgb = jax.nn.sigmoid(rgb)
    # print(raw.dtype, sigma_a.dtype, rgb.dtype, z_vals.dtype)

    # Do volume rendering
    dists = np.concatenate([z_vals[..., 1:] - z_vals[..., :-1], np.broadcast_to([1e10], z_vals[..., :1].shape).astype(rays.dtype)], -1)
    alpha = 1. - np.exp(-sigma_a * dists)
    trans = np.minimum(1., 1. - alpha + 1e-10)
    trans = np.concatenate([np.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    weights = alpha * np.cumprod(trans, -1)
    # print(dists.dtype, alpha.dtype, trans.dtype, weights.dtype)

    rgb_map = np.sum(weights[..., None] * rgb, -2)
    if not allret:
        return rgb_map

    acc_map = np.sum(weights, -1)
    depth_map = np.sum(weights * z_vals, -1)
    return rgb_map, depth_map, acc_map

def CLIPProcessor(image):
    '''
        jax-based preprocessing for CLIP

        image  [B, 3, H, W]: batch image
        return [B, 3, 224, 224]: pre-processed image for CLIP
    '''
    B,D,H,W = image.shape
    image = jax.image.resize(image, (B,D,224,224), 'bicubic') # assume that images have rectangle shape. 
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1,3,1,1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1,3,1,1)
    image = (image - mean.astype(image.dtype)) / std.astype(image.dtype) 
    return image

def SC_loss(rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l):
    '''
        target_emb [1, D]: pre-computed target embedding vector \phi(I)
        source_img [1, 3, H, W]: source image \hat{I}
        l: loss weight lambda
        return: SC_loss
    '''
    # _,H,W,D = rays.shape
    rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l = my_policy.cast_to_compute((rng_inputs, model, params, bds, rays, N_samples, target_emb, CLIP_model, l))
    _,H,W,_ = rays.shape
    source_img = np.clip(render_rays(rng_inputs, model, params, None, np.reshape(rays, (2, -1, 3)), bds[0], bds[1], 1, rand=False), 0, 1)
    source_img = np.reshape(source_img, [1,H,W,3]).transpose(0,3,1,2)
    source_img = CLIPProcessor(source_img)
    source_emb = CLIP_model.get_image_features(pixel_values=source_img)
    source_emb /= np.linalg.norm(source_emb, axis=-1, keepdims=True)
    return l/2 * np.sum((source_emb - target_emb)**2)/source_emb.shape[0]

def single_step_wojit(rng, step, image, rays, params, bds, inner_step_size, N_samples, model, random_ray, target_emb, CLIP_model, K):
    def sgd(param, update):
        return param - inner_step_size * update

    rng, rng_inputs = jax.random.split(rng)

    def loss_model(params):
        g = render_rays(rng_inputs, model, params, None, rays, bds[0], bds[1], N_samples, rand=True)
        L = mse_fn(g, image)
        L = jax.lax.cond(step%K == 0,
            lambda _: L + SC_loss(rng_inputs, model, params, bds, random_ray, N_samples, target_emb, CLIP_model, 1), # exact value of lambda is unknown.
            lambda _: L, 
            operand=None
        )
        return L

    model_loss, grad = jax.value_and_grad(loss_model)(params)
    new_params = jax.tree_multimap(sgd, params, grad)
    return rng, new_params, model_loss

# nn.linen.Module is not jittable
single_step = jit(single_step_wojit, static_argnums=[7, 8, 11])
# optimize render_fn_inner by JIT (func in, func out)
render_fn_inner = jit(render_fn_inner, static_argnums=(1, 7, 8, 9))
mse_fn = jit(lambda x, y: np.mean((x - y)**2))
psnr_fn = jit(lambda x, y: -10 * np.log10(mse_fn(x, y)))

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype = np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype = np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype = np.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def random_pose(rng, bds):
    rng, *rng_inputs = jax.random.split(rng, 3)
    radius = random.uniform(rng_inputs[1], minval = bds[0], maxval = bds[1])
    theta = random.uniform(rng_inputs[1], minval = 0, maxval = 2*np.pi)
    phi = random.uniform(rng_inputs[1], minval = 0, maxval = np.pi/2)
    return pose_spherical(radius,theta,phi)
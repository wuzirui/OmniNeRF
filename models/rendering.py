import torch
import numpy as np
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                c2w_array,
                pose_correction=None,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                use_sdf=True,
                truncation=0.05,
                omni_dir=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
        truncation: the truncation distance of the TSDF field
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """


    def sdf2weights(sdf):
        """
        In neural-rgbd, we didn't change the network architecture of NeRF, but use the
        the output density field of NeRF as sdf value

        Inputs:
            sdf: (N_rays, N_samples)
        Outputs:
            weights: (N_rays, N_samples)
        """
        assert use_sdf, "TSDF is turned off"
        # compute raw weights according to the paper
        weights = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)
        if omni_dir:
            return weights
        # if there exists multiple surface, we should only keep the first one
        # compute the zero-crossing
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        N_rays_ = xyz.shape[0]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c')
        # (N_rays * N_samples_, 3)
        c2ws = c2w_array.reshape(-1, 3, 4).expand(N_samples_, -1, 3, 4).reshape(-1, 3, 4)  # (N_rays_ * N_samples_, 3, 4)
        frame_idx = idx.long().expand(N_samples_, -1, 1).reshape(-1)                     # (N_rays * N_samples_, )
        view_dir = kwargs.get('view_dir', rays_d).expand(N_samples_, -1, 1, 3).reshape(-1, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []

        xyz_ = torch.sum(xyz_[..., None, :] * c2ws[..., :3, :3], axis=-1) + c2ws[..., :3, 3]
        view_dir = torch.sum(view_dir[..., None, :] * c2ws[..., :3, :3], axis=-1)

        if pose_correction is not None:
            R = pose_correction.get_rotation_matrices(frame_idx)
            t = pose_correction.get_translations(frame_idx)
            xyz_ = torch.sum(xyz_[..., None, :] * R, axis=-1) + t
            view_dir = torch.sum(view_dir[..., None, :] * R, axis=-1)

        if typ=='coarse' and test_time and 'fine' in models:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            if omni_dir:
                sigmas_corrs = rearrange(out, '(n1 n2) 2 -> n1 n2 2', n1=N_rays, n2=N_samples_)
                sigmas = sigmas_corrs[..., 0]
                corrs = sigmas_corrs[..., 1]
            else:
                sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2 1', n1=N_rays, n2=N_samples_)

            if use_sdf:
                weights = sdf2weights(sigmas)
        else:
            # infer rgb and sigma and others
            dir_embedded_ = embedding_dir(view_dir)
            # (N_rays * N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded_[i:i+chunk]], 1)
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            # out = out.view(N_rays, N_samples_, 4)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=5 if omni_dir else 4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            if omni_dir:
                corrs = out[..., 4]
            if use_sdf:
                weights = sdf2weights(sigmas)
            
        if not use_sdf:
            # Convert these values using volume rendering (Section 4)
            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

            # compute alpha by the formula (3)
            noise = torch.randn_like(sigmas) * noise_std
            alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

            alphas_shifted = \
                torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
            weights = \
                alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        results[f'sigmas_{typ}'] = sigmas
        if omni_dir:
            results[f'corrs_{typ}'] = corrs
        if test_time and typ == 'coarse' and 'fine' in models:
            return results

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return results

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    idx = rays[:, 8:9] # (N_rays, 1)
    # Embed direction
    # dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        model = models['fine'] if 'fine' in models else models['coarse']
        inference(results, model, 'fine', xyz_fine, z_vals, test_time, **kwargs)

    return results
